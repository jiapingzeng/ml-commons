/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import static org.opensearch.ml.common.utils.StringUtils.gson;

import java.time.Clock;
import java.time.ZoneOffset;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.admin.indices.mapping.get.GetMappingsRequest;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.tasks.TaskCancelledException;
import org.opensearch.core.tasks.TaskId;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.ml.action.agenticindex.IndexProfile.FieldKind;
import org.opensearch.ml.action.agenticindex.LlmOutput.Completion;
import org.opensearch.ml.action.agenticindex.SubQuery.SearchMode;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.indexInsight.MLIndexInsightType;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexRequest;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexRequest.Mode;
import org.opensearch.ml.common.transport.indexInsight.MLIndexInsightGetAction;
import org.opensearch.ml.common.transport.indexInsight.MLIndexInsightGetRequest;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.search.SearchHit;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.search.sort.SortOrder;
import org.opensearch.tasks.CancellableTask;
import org.opensearch.transport.client.Client;

import lombok.extern.log4j.Log4j2;

/**
 * Runs one ask or retrieve request: profile the index, plan parallel subqueries, retrieve, judge sufficiency and
 * issue follow-ups for a bounded number of rounds, then (for ask) answer with passage citations.
 * Searches and LLM calls run with the caller's permissions and as children of the request's task. The deadline and task
 * cancellation are checked before each step; cancelling the task also cancels in-flight child searches, while an
 * in-flight LLM call (prediction tasks are not cancellable) runs to completion.
 */
@Log4j2
public class AgenticIndexRunner {

    static final int PLAN_MAX_TOKENS = 1024;
    static final int JUDGE_MAX_TOKENS = 1024;
    static final int ANSWER_MAX_TOKENS = 1536;
    static final int MAX_PASSAGES = 40;
    private static final int RRF_K = 60;
    static final String NO_EVIDENCE_ANSWER = "The index does not contain passages that answer this question.";

    private final MLAgenticIndexRequest request;
    private final Client client;
    private final Clock clock;
    private final IndexProfileCache profileCache;
    private final String caller;
    private final CancellableTask task;
    private final TaskId parentTaskId;
    private final boolean semanticAvailable;
    private final long startNanos = System.nanoTime();
    private final long deadlineNanos;
    private final List<Map<String, Object>> llmCalls = new ArrayList<>();
    private final List<Map<String, Object>> rounds = new ArrayList<>();
    private final List<String> warnings = new ArrayList<>();
    private final List<SubQuery> executed = new ArrayList<>();

    private IndexProfile profile;
    private List<String> textFields;
    private Set<String> excludedFields;
    private PassageStore passages;
    private String now;
    private Map<String, Object> lastJudgment = Map.of();
    // Round of the last sufficiency judgment; passages from later rounds have not been judged.
    private int judgedRound;
    private boolean timedOut;

    /**
     * @param caller       identity of the calling user, part of the profile cache key
     * @param timeout      deadline of the whole request
     * @param task         the request's task, checked for cancellation between steps
     * @param localNodeId  id of the node running the task, used to make searches and predictions its children
     */
    public AgenticIndexRunner(
        MLAgenticIndexRequest request,
        Client client,
        Clock clock,
        IndexProfileCache profileCache,
        String caller,
        TimeValue timeout,
        CancellableTask task,
        String localNodeId
    ) {
        this.request = request;
        this.client = client;
        this.clock = clock;
        this.profileCache = profileCache;
        this.caller = caller;
        this.task = task;
        this.parentTaskId = new TaskId(localNodeId, task.getId());
        this.deadlineNanos = startNanos + timeout.nanos();
        this.semanticAvailable = request.getEmbeddingModelId() != null;
    }

    public void run(ActionListener<Map<String, Object>> listener) {
        now = DateTimeFormatter.ISO_OFFSET_DATE_TIME.format(clock.instant().atOffset(ZoneOffset.UTC).withNano(0));
        buildProfile(ActionListener.wrap(p -> {
            if (!stopped("planning", listener)) {
                plan(listener);
            }
        }, listener::onFailure));
    }

    /**
     * Fails the listener when the task was cancelled, and finishes with partial results once the deadline has passed.
     * Returns true when the caller must not start the next step.
     */
    private boolean stopped(String nextStep, ActionListener<Map<String, Object>> listener) {
        if (task.isCancelled()) {
            listener.onFailure(new TaskCancelledException("The agentic index request was cancelled: " + task.getReasonCancelled()));
            return true;
        }
        if (deadlinePassed(nextStep)) {
            finish(listener);
            return true;
        }
        return false;
    }

    private boolean deadlinePassed(String nextStep) {
        if (!timedOut && System.nanoTime() - deadlineNanos >= 0) {
            timedOut = true;
            warnings.add("request deadline reached before " + nextStep + "; returned partial results");
        }
        return timedOut;
    }

    private TimeValue remainingTime() {
        return TimeValue.timeValueNanos(Math.max(TimeValue.timeValueMillis(1).nanos(), deadlineNanos - System.nanoTime()));
    }

    @SuppressWarnings("unchecked")
    void buildProfile(ActionListener<IndexProfile> listener) {
        client.admin().indices().getMappings(new GetMappingsRequest().indices(request.getIndex()), ActionListener.wrap(response -> {
            Map<String, MappingMetadata> mappings = response.getMappings();
            if (mappings == null || mappings.isEmpty()) {
                listener.onFailure(new IllegalArgumentException("Index [" + request.getIndex() + "] has no mapping"));
                return;
            }
            if (mappings.size() > 1) {
                listener
                    .onFailure(
                        new OpenSearchStatusException(
                            String
                                .format(
                                    Locale.ROOT,
                                    "[%s] resolves to %d indices %s. The agentic index APIs query one index; "
                                        + "pass a single index name or an alias that points to one index.",
                                    request.getIndex(),
                                    mappings.size(),
                                    new TreeSet<>(mappings.keySet())
                                ),
                            RestStatus.BAD_REQUEST
                        )
                    );
                return;
            }
            String concreteIndex = mappings.keySet().iterator().next();
            String cacheKey = profileCache.key(caller, request.getIndex(), concreteIndex, request.getTextFields());
            IndexProfile cached = profileCache.get(cacheKey);
            if (cached != null) {
                onProfile(cached, listener);
                return;
            }
            MappingMetadata mapping = mappings.get(concreteIndex);
            Map<String, Object> properties = (Map<String, Object>) mapping.getSourceAsMap().get("properties");
            IndexProfile fresh = IndexProfile.fromMappingProperties(request.getIndex(), properties, request.getTextFields());
            SearchSourceBuilder statsQuery = fresh.statsQuery().timeout(remainingTime());
            client.search(childSearch(statsQuery), ActionListener.wrap(stats -> {
                fresh.applyStats(stats.getAggregations());
                if (stats.isTimedOut() || stats.getFailedShards() > 0) {
                    warnings.add("index field statistics are incomplete (timed out or failed shards); planned with partial field values");
                } else {
                    profileCache.put(cacheKey, fresh);
                }
                onProfile(fresh, listener);
            }, e -> {
                // Field values only improve filter planning, so the request continues without them.
                log.warn("Agentic index statistics search failed on [{}]: {}", request.getIndex(), e.getMessage());
                warnings.add("index field statistics unavailable (" + e.getMessage() + "); planned without field values");
                onProfile(fresh, listener);
            }));
        }, listener::onFailure));
    }

    private void onProfile(IndexProfile base, ActionListener<IndexProfile> listener) {
        profile = base;
        textFields = request.getTextFields().isEmpty() ? profile.profiledFieldsOfKind(FieldKind.TEXT) : request.getTextFields();
        if (textFields.isEmpty()) {
            listener.onFailure(new IllegalArgumentException("Index [" + request.getIndex() + "] has no text fields to search"));
            return;
        }
        excludedFields = new HashSet<>(profile.fieldsOfKind(FieldKind.VECTOR));
        if (request.getEmbeddingField() != null) {
            excludedFields.add(request.getEmbeddingField());
        }
        passages = new PassageStore(textFields, excludedFields, MAX_PASSAGES);
        if (!request.isUseIndexInsight()) {
            listener.onResponse(profile);
            return;
        }
        MLIndexInsightGetRequest insightRequest = MLIndexInsightGetRequest
            .builder()
            .indexName(request.getIndex())
            .targetIndexInsight(MLIndexInsightType.FIELD_DESCRIPTION)
            .tenantId(request.getTenantId())
            .build();
        client.execute(MLIndexInsightGetAction.INSTANCE, insightRequest, ActionListener.wrap(insight -> {
            profile = profile.withIndexInsight(insight.getIndexInsight().getContent());
            listener.onResponse(profile);
        }, e -> {
            log.debug("Agentic index could not read index insight", e);
            warnings.add("index insight unavailable (" + e.getMessage() + "); planned without it");
            listener.onResponse(profile);
        }));
    }

    private void plan(ActionListener<Map<String, Object>> listener) {
        String prompt = AgenticIndexPrompts
            .planner(profile.render(), request.getQuestion(), now, request.getMaxSubqueries(), semanticAvailable);
        callLlm("plan", prompt, PLAN_MAX_TOKENS, ActionListener.wrap(completion -> {
            List<SubQuery> subQueries = parseSubQueries(completion.content(), "subqueries", "plan");
            if (subQueries.isEmpty()) {
                warnings.add("planner returned no usable subqueries; searched the question text");
                subQueries = List
                    .of(
                        new SubQuery(request.getQuestion(), semanticAvailable ? SearchMode.HYBRID : SearchMode.KEYWORD, Map.of(), List.of())
                    );
            }
            executeRound(1, subQueries, listener);
        }, listener::onFailure));
    }

    @SuppressWarnings("unchecked")
    List<SubQuery> parseSubQueries(String content, String key, String step) {
        List<SubQuery> result = new ArrayList<>();
        Object items;
        try {
            items = LlmOutput.extractJsonObject(content).get(key);
        } catch (IllegalArgumentException e) {
            warnings.add(step + " output was not JSON");
            return result;
        }
        if (items instanceof List) {
            for (Object item : (List<Object>) items) {
                if (result.size() >= request.getMaxSubqueries()) {
                    warnings.add(step + " proposed more than " + request.getMaxSubqueries() + " subqueries; extra ones dropped");
                    break;
                }
                SubQuery subQuery = SubQuery.fromPlan(item, profile, semanticAvailable);
                if (subQuery != null && !alreadyExecuted(subQuery)) {
                    result.add(subQuery);
                }
            }
        }
        return result;
    }

    private boolean alreadyExecuted(SubQuery candidate) {
        for (SubQuery done : executed) {
            if (done.getText().equalsIgnoreCase(candidate.getText()) && done.getFilters().equals(candidate.getFilters())) {
                return true;
            }
        }
        return false;
    }

    void executeRound(int round, List<SubQuery> subQueries, ActionListener<Map<String, Object>> listener) {
        if (stopped("search round " + round, listener)) {
            return;
        }
        long roundStart = System.nanoTime();
        executed.addAll(subQueries);
        AtomicReferenceArray<List<SearchHit>> hits = new AtomicReferenceArray<>(subQueries.size());
        AtomicReferenceArray<String> errors = new AtomicReferenceArray<>(subQueries.size());
        AtomicInteger pending = new AtomicInteger(subQueries.size());
        if (subQueries.isEmpty()) {
            afterRound(round, subQueries, hits, errors, roundStart, listener);
            return;
        }
        for (int i = 0; i < subQueries.size(); i++) {
            int idx = i;
            runSubQuery(subQueries.get(i), ActionListener.wrap(result -> {
                hits.set(idx, result);
                if (pending.decrementAndGet() == 0) {
                    afterRound(round, subQueries, hits, errors, roundStart, listener);
                }
            }, e -> {
                // A model-planned subquery can be invalid (for example a bad date); report it and keep the others.
                log.debug("Agentic index subquery failed", e);
                errors.set(idx, e.getMessage());
                hits.set(idx, List.of());
                if (pending.decrementAndGet() == 0) {
                    afterRound(round, subQueries, hits, errors, roundStart, listener);
                }
            }));
        }
    }

    private void afterRound(
        int round,
        List<SubQuery> subQueries,
        AtomicReferenceArray<List<SearchHit>> hits,
        AtomicReferenceArray<String> errors,
        long roundStart,
        ActionListener<Map<String, Object>> listener
    ) {
        try {
            List<Map<String, Object>> subQueryTrace = new ArrayList<>();
            for (int i = 0; i < subQueries.size(); i++) {
                List<String> ids = new ArrayList<>();
                for (SearchHit hit : hits.get(i)) {
                    String id = passages.add(hit, round, executed.size() - subQueries.size() + i);
                    if (id != null) {
                        ids.add(id);
                    }
                }
                Map<String, Object> entry = subQueries.get(i).toMap();
                entry.put("passages", ids);
                if (errors.get(i) != null) {
                    entry.put("error", errors.get(i));
                }
                subQueryTrace.add(entry);
            }
            Map<String, Object> roundTrace = new LinkedHashMap<>();
            roundTrace.put("round", round);
            roundTrace.put("subqueries", subQueryTrace);
            roundTrace.put("search_took_ms", millisSince(roundStart));
            rounds.add(roundTrace);
            judge(round, roundTrace, listener);
        } catch (Exception e) {
            listener.onFailure(e);
        }
    }

    @SuppressWarnings("unchecked")
    private void judge(int round, Map<String, Object> roundTrace, ActionListener<Map<String, Object>> listener) {
        if (stopped("sufficiency check of round " + round, listener)) {
            return;
        }
        if (passages.size() == 0 && round >= request.getMaxRounds()) {
            lastJudgment = Map.of("sufficient", false, "missing", "no passages retrieved");
            finish(listener);
            return;
        }
        List<Map<String, Object>> previous = new ArrayList<>();
        executed.forEach(q -> previous.add(q.toMap()));
        String prompt = AgenticIndexPrompts
            .sufficiency(
                profile.render(),
                request.getQuestion(),
                now,
                passages.render(passages.all()),
                gson.toJson(previous),
                request.getMaxSubqueries(),
                semanticAvailable
            );
        callLlm("sufficiency_round_" + round, prompt, JUDGE_MAX_TOKENS, ActionListener.wrap(completion -> {
            Map<String, Object> judgment;
            try {
                judgment = LlmOutput.extractJsonObject(completion.content());
            } catch (IllegalArgumentException e) {
                warnings.add("sufficiency output was not JSON in round " + round + "; treated as sufficient");
                judgment = Map.of("sufficient", true);
            }
            boolean sufficient = Boolean.TRUE.equals(judgment.get("sufficient"));
            Map<String, Object> summary = new LinkedHashMap<>();
            summary.put("sufficient", sufficient);
            // An explicit relevant list (even empty) selects the passages; without one, all passages are kept.
            if (judgment.get("relevant") instanceof List) {
                summary.put("relevant", validPassageIds(LlmOutput.passageIdList(judgment.get("relevant"))));
            }
            summary.put("missing", judgment.getOrDefault("missing", ""));
            roundTrace.put("judgment", summary);
            lastJudgment = summary;
            judgedRound = round;
            if (sufficient || round >= request.getMaxRounds()) {
                finish(listener);
                return;
            }
            List<SubQuery> followups = parseSubQueries(completion.content(), "followups", "sufficiency");
            if (followups.isEmpty()) {
                warnings.add("evidence judged insufficient in round " + round + " but no new follow-up subqueries were proposed");
                finish(listener);
                return;
            }
            executeRound(round + 1, followups, listener);
        }, listener::onFailure));
    }

    private List<String> validPassageIds(List<String> ids) {
        List<String> valid = new ArrayList<>();
        for (String id : ids) {
            if (passages.get(id) != null && !valid.contains(id)) {
                valid.add(id);
            }
        }
        return valid;
    }

    // The last judgment's relevant passages, plus any passages retrieved after it (a round cut off by the deadline).
    @SuppressWarnings("unchecked")
    private List<Map<String, Object>> selectedPassages() {
        List<Map<String, Object>> selected = new ArrayList<>();
        if (lastJudgment.get("relevant") instanceof List) {
            ((List<String>) lastJudgment.get("relevant")).forEach(id -> selected.add(passages.get(id)));
            passages.all().stream().filter(p -> (Integer) p.get("round") > judgedRound).forEach(selected::add);
        } else {
            selected.addAll(passages.all());
        }
        return selected;
    }

    private void finish(ActionListener<Map<String, Object>> listener) {
        List<Map<String, Object>> selected = selectedPassages();
        if (request.getMode() == Mode.ASK && !selected.isEmpty() && !timedOut) {
            if (task.isCancelled()) {
                listener.onFailure(new TaskCancelledException("The agentic index request was cancelled: " + task.getReasonCancelled()));
                return;
            }
            deadlinePassed("answer");
        }
        if (timedOut) {
            Map<String, Object> result = baseResult();
            if (request.getMode() == Mode.ASK) {
                result.put("answer", null);
            }
            result.put("passages", selected);
            result.put("trace", trace());
            listener.onResponse(result);
            return;
        }
        if (request.getMode() == Mode.RETRIEVE || selected.isEmpty()) {
            Map<String, Object> result = baseResult();
            if (request.getMode() == Mode.ASK) {
                result.put("answer", NO_EVIDENCE_ANSWER);
                result.put("citations", List.of());
            }
            result.put("passages", selected);
            result.put("trace", trace());
            listener.onResponse(result);
            return;
        }
        List<String> labels = new ArrayList<>();
        Map<String, String> labelToId = new LinkedHashMap<>();
        for (Map<String, Object> passage : selected) {
            String label = "P" + (labels.size() + 1);
            labels.add(label);
            labelToId.put(label, (String) passage.get("passage_id"));
        }
        callLlm(
            "answer",
            AgenticIndexPrompts.answer(request.getQuestion(), now, passages.render(selected, labels)),
            ANSWER_MAX_TOKENS,
            ActionListener.wrap(completion -> {
                List<Map<String, Object>> citations = new ArrayList<>();
                List<String> invalid = new ArrayList<>();
                for (String label : LlmOutput.citedPassageIds(completion.content())) {
                    if (labelToId.containsKey(label)) {
                        citations.add(passages.citation(labelToId.get(label)));
                    } else {
                        invalid.add(label);
                    }
                }
                if (!invalid.isEmpty()) {
                    warnings.add("answer cited passages that were not provided " + invalid + "; markers removed");
                }
                String answer = LlmOutput.relabelCitations(completion.content(), labelToId);
                if (citations.isEmpty()) {
                    warnings.add("answer has no citations");
                }
                Map<String, Object> result = baseResult();
                result.put("answer", answer);
                result.put("citations", citations);
                result.put("passages", selected);
                result.put("trace", trace());
                listener.onResponse(result);
            }, listener::onFailure)
        );
    }

    private Map<String, Object> baseResult() {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("mode", request.getMode().name().toLowerCase(Locale.ROOT));
        result.put("question", request.getQuestion());
        result.put("sufficient", lastJudgment.getOrDefault("sufficient", false));
        result.put("timed_out", timedOut);
        Object missing = lastJudgment.get("missing");
        if (missing != null && !String.valueOf(missing).isBlank()) {
            result.put("missing", missing);
        }
        return result;
    }

    private Map<String, Object> trace() {
        long promptTokens = 0;
        long completionTokens = 0;
        for (Map<String, Object> call : llmCalls) {
            promptTokens += (Long) call.get("prompt_tokens");
            completionTokens += (Long) call.get("completion_tokens");
        }
        Map<String, Object> trace = new LinkedHashMap<>();
        trace.put("took_ms", millisSince(startNanos));
        trace.put("llm_call_count", llmCalls.size());
        trace.put("prompt_tokens", promptTokens);
        trace.put("completion_tokens", completionTokens);
        trace.put("llm_calls", llmCalls);
        trace.put("rounds", rounds);
        if (!warnings.isEmpty()) {
            trace.put("warnings", warnings);
        }
        return trace;
    }

    void runSubQuery(SubQuery subQuery, ActionListener<List<SearchHit>> listener) {
        ActionListener<List<SearchHit>> once = ActionListener.notifyOnce(listener);
        try {
            int k = request.getK();
            boolean hasText = !subQuery.getText().isEmpty();
            if (!hasText || subQuery.getMode() == SearchMode.KEYWORD) {
                search(subQuery.keywordQuery(textFields), k, !hasText ? sortField(subQuery) : null, once);
                return;
            }
            QueryBuilder semantic = subQuery.semanticQuery(request.getEmbeddingField(), request.getEmbeddingModelId(), k);
            if (subQuery.getMode() == SearchMode.SEMANTIC) {
                search(semantic, k, null, once);
                return;
            }
            // Hybrid: keyword and semantic searches run in parallel and are fused when both return.
            AtomicReferenceArray<List<SearchHit>> results = new AtomicReferenceArray<>(2);
            AtomicInteger remaining = new AtomicInteger(2);
            List<QueryBuilder> queries = List.of(subQuery.keywordQuery(textFields), semantic);
            for (int i = 0; i < queries.size(); i++) {
                int idx = i;
                search(queries.get(i), k, null, ActionListener.wrap(result -> {
                    results.set(idx, result);
                    if (remaining.decrementAndGet() == 0) {
                        once.onResponse(reciprocalRankFusion(results.get(0), results.get(1), k));
                    }
                }, once::onFailure));
            }
        } catch (Exception e) {
            once.onFailure(e);
        }
    }

    // Filter-only subqueries ("latest pricing update") return the newest documents first.
    private String sortField(SubQuery subQuery) {
        for (String field : subQuery.getFilters().keySet()) {
            if (profile.kindOf(field) == FieldKind.DATE && profile.hasDocValues(field)) {
                return field;
            }
        }
        return profile.fieldsOfKind(FieldKind.DATE).stream().filter(profile::hasDocValues).findFirst().orElse(null);
    }

    private void search(QueryBuilder query, int k, String sortField, ActionListener<List<SearchHit>> listener) {
        SearchSourceBuilder source = new SearchSourceBuilder().query(query).size(k).trackTotalHits(false).timeout(remainingTime());
        source.fetchSource(sourceIncludes(), excludedFields.toArray(new String[0]));
        if (sortField != null) {
            source.sort(sortField, SortOrder.DESC);
        }
        client.search(childSearch(source), ActionListener.wrap((SearchResponse response) -> {
            listener.onResponse(List.of(response.getHits().getHits()));
        }, listener::onFailure));
    }

    // Only fields a passage can use are fetched, so wide documents stay cheap: text, profiled and location fields.
    String[] sourceIncludes() {
        Set<String> includes = new LinkedHashSet<>(textFields);
        includes.addAll(profile.getProfiledFields());
        includes.add("title");
        includes.addAll(PassageStore.LOCATION_FIELDS);
        return includes.toArray(new String[0]);
    }

    private SearchRequest childSearch(SearchSourceBuilder source) {
        SearchRequest searchRequest = new SearchRequest(request.getIndex()).source(source);
        searchRequest.setParentTask(parentTaskId);
        return searchRequest;
    }

    static List<SearchHit> reciprocalRankFusion(List<SearchHit> first, List<SearchHit> second, int k) {
        Map<String, SearchHit> hitsByKey = new LinkedHashMap<>();
        Map<String, Double> scores = new LinkedHashMap<>();
        for (List<SearchHit> list : List.of(first, second)) {
            for (int rank = 0; rank < list.size(); rank++) {
                SearchHit hit = list.get(rank);
                String key = hit.getIndex() + "/" + hit.getId();
                hitsByKey.putIfAbsent(key, hit);
                scores.merge(key, 1.0 / (RRF_K + rank + 1), Double::sum);
            }
        }
        List<String> keys = new ArrayList<>(scores.keySet());
        keys.sort((a, b) -> Double.compare(scores.get(b), scores.get(a)));
        List<SearchHit> fused = new ArrayList<>();
        for (String key : keys.subList(0, Math.min(k, keys.size()))) {
            SearchHit hit = hitsByKey.get(key);
            hit.score(scores.get(key).floatValue());
            fused.add(hit);
        }
        return fused;
    }

    void callLlm(String step, String userPrompt, int maxTokens, ActionListener<Completion> listener) {
        List<Map<String, String>> messages = List
            .of(Map.of("role", "system", "content", AgenticIndexPrompts.SYSTEM), Map.of("role", "user", "content", userPrompt));
        Map<String, String> parameters = new LinkedHashMap<>();
        parameters.put("messages", gson.toJson(messages));
        parameters.put("max_tokens", String.valueOf(maxTokens));
        MLPredictionTaskRequest predictRequest = MLPredictionTaskRequest
            .builder()
            .modelId(request.getLlmModelId())
            .mlInput(
                MLInput
                    .builder()
                    .algorithm(FunctionName.REMOTE)
                    .inputDataset(RemoteInferenceInputDataSet.builder().parameters(parameters).build())
                    .build()
            )
            .tenantId(request.getTenantId())
            .build();
        predictRequest.setParentTask(parentTaskId);
        long callStart = System.nanoTime();
        client.execute(MLPredictionTaskAction.INSTANCE, predictRequest, ActionListener.wrap(response -> {
            Completion completion = LlmOutput.fromModelOutput(request.getLlmModelId(), response.getOutput());
            Map<String, Object> call = new LinkedHashMap<>();
            call.put("step", step);
            call.put("took_ms", millisSince(callStart));
            call.put("prompt_tokens", completion.promptTokens());
            call.put("completion_tokens", completion.completionTokens());
            llmCalls.add(call);
            listener.onResponse(completion);
        }, listener::onFailure));
    }

    private static long millisSince(long startNanos) {
        return (System.nanoTime() - startNanos) / 1_000_000;
    }
}
