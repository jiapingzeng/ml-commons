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
import java.util.Objects;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.function.Consumer;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.StepListener;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.tasks.TaskCancelledException;
import org.opensearch.core.tasks.TaskId;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.ml.action.agenticindex.BoundedFanOut.Outcome;
import org.opensearch.ml.action.agenticindex.IndexProfile.FieldKind;
import org.opensearch.ml.action.agenticindex.IndexResolver.ResolvedIndex;
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
 * Runs one ask or retrieve request: resolve the target to the indices the caller can read, profile each index from its
 * mapping, and, when there are several, route the question to the relevant ones from a compact catalog. Field
 * statistics are collected only for the selected indices. Then plan parallel
 * subqueries (each on one index), retrieve, judge sufficiency and issue follow-ups for a bounded number of rounds, and
 * (for ask) answer with passage citations.
 * Mapping reads, searches and LLM calls run with the caller's permissions and as children of the request's task. The
 * deadline and task cancellation are checked before each step; cancelling the task also cancels in-flight child
 * searches, while an in-flight LLM call (prediction tasks are not cancellable) runs to completion.
 */
@Log4j2
public class AgenticIndexRunner {

    static final int PLAN_MAX_TOKENS = 1024;
    static final int ROUTE_MAX_TOKENS = 1024;
    static final int JUDGE_MAX_TOKENS = 1024;
    static final int ANSWER_MAX_TOKENS = 1536;
    static final int MAX_PASSAGES = 40;
    static final int PROFILE_CONCURRENCY = 8;
    // Budget for the selected indices' profiles in one prompt, shared evenly but at least MIN_PROFILE_CHARS each.
    static final int PROFILES_MAX_CHARS = 24000;
    static final int MIN_PROFILE_CHARS = 1500;
    // Budget for the catalog of unselected indices in the sufficiency prompt.
    static final int OTHER_INDICES_MAX_CHARS = 4000;
    private static final int RRF_K = 60;
    static final String NO_EVIDENCE_ANSWER = "The index does not contain passages that answer this question.";

    /** Search state of one profiled index. */
    static final class SearchableIndex {
        final ResolvedIndex resolved;
        IndexProfile profile;
        final List<String> textFields;
        final Set<String> excludedFields;
        // Vector field for semantic search, or null when this index cannot be searched semantically.
        final String embeddingField;
        // Whether field statistics were added or attempted; only selected indices get them.
        boolean statsRequested;
        // Whether index insight was requested, when the request uses it.
        boolean insightRequested;

        SearchableIndex(
            ResolvedIndex resolved,
            IndexProfile profile,
            List<String> textFields,
            Set<String> excludedFields,
            String embeddingField
        ) {
            this.resolved = resolved;
            this.profile = profile;
            this.textFields = textFields;
            this.excludedFields = excludedFields;
            this.embeddingField = embeddingField;
        }

        String name() {
            return resolved.name();
        }

        boolean semanticAvailable() {
            return embeddingField != null;
        }
    }

    private final MLAgenticIndexRequest request;
    private final Client client;
    private final Clock clock;
    private final IndexProfileCache profileCache;
    private final IndexResolver resolver;
    private final int maxIndices;
    private final String caller;
    private final CancellableTask task;
    private final TaskId parentTaskId;
    private final long startNanos = System.nanoTime();
    private final long deadlineNanos;
    private final List<Map<String, Object>> llmCalls = new ArrayList<>();
    private final List<Map<String, Object>> rounds = new ArrayList<>();
    private final List<String> warnings = new ArrayList<>();
    private final List<SubQuery> executed = new ArrayList<>();
    // Profiled indices by concrete name, in name order.
    private final Map<String, SearchableIndex> indices = new LinkedHashMap<>();
    // Indices the planner may search: the routed ones, plus any a follow-up round added.
    private final Set<String> selected = new LinkedHashSet<>();
    private final List<Map<String, Object>> selectedTrace = new ArrayList<>();
    private final Map<String, Object> routingTrace = new LinkedHashMap<>();

    private boolean multi;
    private int resolvedCount;
    private PassageStore passages;
    private String now;
    private Map<String, Object> lastJudgment = Map.of();
    // Round of the last sufficiency judgment; passages from later rounds have not been judged.
    private int judgedRound;
    private boolean timedOut;

    /**
     * @param resolver     resolves the target to the indices the caller can read
     * @param maxIndices   most indices the target may resolve to
     * @param caller       identity of the calling user, part of the profile cache key
     * @param timeout      deadline of the whole request
     * @param task         the request's task, checked for cancellation between steps
     * @param localNodeId  id of the node running the task, used to make child requests its children
     */
    public AgenticIndexRunner(
        MLAgenticIndexRequest request,
        Client client,
        Clock clock,
        IndexProfileCache profileCache,
        IndexResolver resolver,
        int maxIndices,
        String caller,
        TimeValue timeout,
        CancellableTask task,
        String localNodeId
    ) {
        this.request = request;
        this.client = client;
        this.clock = clock;
        this.profileCache = profileCache;
        this.resolver = resolver;
        this.maxIndices = maxIndices;
        this.caller = caller;
        this.task = task;
        this.parentTaskId = new TaskId(localNodeId, task.getId());
        this.deadlineNanos = startNanos + timeout.nanos();
    }

    public void run(ActionListener<Map<String, Object>> listener) {
        now = DateTimeFormatter.ISO_OFFSET_DATE_TIME.format(clock.instant().atOffset(ZoneOffset.UTC).withNano(0));
        resolveAndProfile(ActionListener.wrap(v -> {
            if (!multi) {
                afterSelection(listener);
            } else if (!stopped("routing", listener)) {
                route(ActionListener.wrap(r -> afterSelection(listener), listener::onFailure));
            }
        }, e -> {
            if (e instanceof IndexResolver.TimedOutException && !task.isCancelled()) {
                deadlinePassed("the target was resolved");
                finish(listener);
            } else {
                listener.onFailure(e);
            }
        }));
    }

    private void afterSelection(ActionListener<Map<String, Object>> listener) {
        addStats(ActionListener.wrap(s -> addIndexInsight(ActionListener.wrap(v -> {
            if (!stopped("planning", listener)) {
                plan(listener);
            }
        }, listener::onFailure)), listener::onFailure));
    }

    /**
     * Fails the listener when the task was cancelled, and finishes with partial results once the deadline has passed.
     * Returns true when the caller must not start the next step.
     */
    private boolean stopped(String nextStep, ActionListener<Map<String, Object>> listener) {
        if (task.isCancelled()) {
            listener.onFailure(cancelled());
            return true;
        }
        if (deadlinePassed(nextStep)) {
            finish(listener);
            return true;
        }
        return false;
    }

    private TaskCancelledException cancelled() {
        return new TaskCancelledException("The agentic index request was cancelled: " + task.getReasonCancelled());
    }

    private boolean deadlinePassed(String nextStep) {
        if (!timedOut && deadlineReached()) {
            timedOut = true;
            warnings.add("request deadline reached before " + nextStep + "; returned partial results");
        }
        return timedOut;
    }

    private boolean deadlineReached() {
        return System.nanoTime() - deadlineNanos >= 0;
    }

    private TimeValue remainingTime() {
        return TimeValue.timeValueNanos(Math.max(TimeValue.timeValueMillis(1).nanos(), deadlineNanos - System.nanoTime()));
    }

    /**
     * Resolves the target and builds a mapping-only profile of every resolved index; field statistics are added later,
     * only for selected indices. A target with one index is selected directly.
     */
    void resolveAndProfile(ActionListener<Void> listener) {
        resolver
            .resolve(
                request.getIndex(),
                maxIndices,
                parentTaskId,
                task::isCancelled,
                this::deadlineReached,
                ActionListener.wrap(resolved -> {
                    resolvedCount = resolved.size();
                    multi = resolved.size() > 1;
                    onResolved(resolved, listener);
                }, listener::onFailure)
            );
    }

    private void onResolved(List<ResolvedIndex> resolved, ActionListener<Void> listener) {
        if (task.isCancelled()) {
            listener.onFailure(cancelled());
            return;
        }
        List<String> skipped = new ArrayList<>();
        for (ResolvedIndex index : resolved) {
            // A single index keeps the requested target as its display name, as before multi-index support.
            IndexProfile profile = IndexProfile
                .fromMappingProperties(multi ? index.name() : request.getIndex(), index.properties(), request.getTextFields())
                .withDescription(index.description());
            List<String> notes = new ArrayList<>();
            SearchableIndex searchable;
            try {
                searchable = searchable(index, profile, notes);
            } catch (IllegalArgumentException e) {
                if (!multi) {
                    listener.onFailure(e);
                    return;
                }
                String reason = "index [" + index.name() + "] skipped: " + e.getMessage();
                warnings.add(reason);
                skipped.add(reason);
                continue;
            }
            warnings.addAll(notes);
            indices.put(index.name(), searchable);
        }
        if (indices.isEmpty()) {
            listener
                .onFailure(
                    new OpenSearchStatusException(
                        String
                            .format(
                                Locale.ROOT,
                                "None of the %d indices that [%s] resolves to can be searched: %s",
                                resolved.size(),
                                request.getIndex(),
                                String.join("; ", skipped.subList(0, Math.min(5, skipped.size())))
                            ),
                        RestStatus.BAD_REQUEST
                    )
                );
            return;
        }
        if (!multi) {
            select(indices.keySet().iterator().next(), null);
        } else {
            routingTrace.put("profiled_index_count", indices.size());
            routingTrace.put("resolve_took_ms", millisSince(startNanos));
        }
        passages = new PassageStore(
            index -> indices.containsKey(index) ? indices.get(index).textFields : List.of(),
            index -> indices.containsKey(index) ? indices.get(index).excludedFields : Set.of(),
            MAX_PASSAGES,
            multi
        );
        listener.onResponse(null);
    }

    // Warning prefix naming the index when the request covers several.
    private String prefix(String index) {
        return multi ? "index [" + index + "]: " : "";
    }

    /** Adds field statistics to the selected indices that do not have them yet, from the cache or a statistics search. */
    private void addStats(ActionListener<Void> listener) {
        List<SearchableIndex> pending = selected.stream().map(indices::get).filter(i -> !i.statsRequested).toList();
        pending.forEach(i -> i.statsRequested = true);
        BoundedFanOut.<SearchableIndex, String>run(pending, PROFILE_CONCURRENCY, this::addStats, outcomes -> {
            for (Outcome<String> outcome : outcomes) {
                if (outcome.result() != null) {
                    warnings.add(outcome.result());
                }
            }
            listener.onResponse(null);
        });
    }

    // Completes with a warning, or null when the statistics were added.
    private void addStats(SearchableIndex index, ActionListener<String> listener) {
        ResolvedIndex resolved = index.resolved;
        String cacheKey = cacheKey(resolved);
        IndexProfile cached = profileCache.get(cacheKey);
        if (cached != null) {
            index.profile.copyStatsFrom(cached);
            listener.onResponse(null);
            return;
        }
        if (task.isCancelled() || deadlineReached()) {
            // The request stops before planning; keep the mapping-only profile.
            listener.onResponse(null);
            return;
        }
        IndexProfile profile = index.profile;
        SearchSourceBuilder statsQuery = profile.statsQuery().timeout(remainingTime());
        client.search(childSearch(resolved, statsQuery), ActionListener.wrap(stats -> {
            profile.applyStats(stats.getAggregations());
            if (stats.isTimedOut() || stats.getFailedShards() > 0) {
                listener
                    .onResponse(
                        prefix(resolved.name())
                            + "index field statistics are incomplete (timed out or failed shards); planned with partial field values"
                    );
                return;
            }
            profileCache.put(cacheKey, profile);
            listener.onResponse(null);
        }, e -> {
            // Field values only improve filter planning, so the request continues without them.
            log.warn("Agentic index statistics search failed on [{}]: {}", resolved.name(), e.getMessage());
            listener
                .onResponse(
                    prefix(resolved.name()) + "index field statistics unavailable (" + e.getMessage() + "); planned without field values"
                );
        }));
    }

    private String cacheKey(ResolvedIndex resolved) {
        return profileCache.key(caller, resolved.target(), resolved.name(), request.getTextFields());
    }

    private SearchableIndex searchable(ResolvedIndex resolved, IndexProfile profile, List<String> notes) {
        List<String> requested = request.getTextFields();
        List<String> textFields;
        if (requested.isEmpty()) {
            textFields = profile.profiledFieldsOfKind(FieldKind.TEXT);
        } else if (multi) {
            // Requested fields apply to each index that has them.
            textFields = requested.stream().filter(f -> profile.kindOf(f) != null).toList();
        } else {
            textFields = requested;
        }
        if (textFields.isEmpty()) {
            String index = multi ? resolved.name() : request.getIndex();
            throw new IllegalArgumentException(
                requested.isEmpty() || !multi
                    ? "Index [" + index + "] has no text fields to search"
                    : "Index [" + index + "] has none of the requested text fields " + requested
            );
        }
        List<String> vectors = profile.fieldsOfKind(FieldKind.VECTOR);
        Set<String> excluded = new HashSet<>(vectors);
        String requestedField = request.getEmbeddingField();
        if (requestedField != null) {
            excluded.add(requestedField);
        }
        String embeddingField = null;
        if (request.getEmbeddingModelId() != null) {
            if (requestedField != null && (!multi || vectors.contains(requestedField))) {
                embeddingField = requestedField;
            } else if (!vectors.isEmpty()) {
                embeddingField = vectors.get(0);
                if (vectors.size() > 1) {
                    notes.add(prefix(resolved.name()) + "several vector fields " + vectors + "; semantic search uses " + embeddingField);
                }
            } else {
                notes.add(prefix(resolved.name()) + "no vector field; searched with keyword only");
            }
        }
        return new SearchableIndex(resolved, profile, textFields, excluded, embeddingField);
    }

    private void select(String index, String reason) {
        if (selected.add(index) && multi) {
            Map<String, Object> entry = new LinkedHashMap<>();
            entry.put("index", index);
            entry.put("reason", reason);
            selectedTrace.add(entry);
        }
    }

    /**
     * Selects up to max_indices_selected indices: all of them when they fit, otherwise the LLM picks from the catalog.
     */
    void route(ActionListener<Void> listener) {
        int max = request.getMaxIndicesSelected();
        if (indices.size() <= max) {
            String reason = String.format(Locale.ROOT, "all %d searchable indices fit max_indices_selected", indices.size());
            indices.keySet().forEach(name -> select(name, reason));
            routingTrace.put("routed", false);
            listener.onResponse(null);
            return;
        }
        long catalogStart = System.nanoTime();
        // Cached field values of earlier requests by the same caller help routing; no statistics search runs here.
        int cachedStats = 0;
        for (SearchableIndex index : indices.values()) {
            IndexProfile cached = profileCache.get(cacheKey(index.resolved));
            if (cached != null) {
                index.profile.copyStatsFrom(cached);
                cachedStats++;
            }
        }
        List<IndexProfile> profiles = indices.values().stream().map(i -> i.profile).toList();
        IndexCatalog.Catalog catalog = IndexCatalog.render(profiles, request.getQuestion(), IndexCatalog.MAX_CHARS);
        routingTrace.put("routed", true);
        routingTrace.put("catalog_took_ms", millisSince(catalogStart));
        routingTrace.put("catalog_chars", catalog.text().length());
        routingTrace.put("catalog_index_count", catalog.indices().size());
        routingTrace.put("catalog_prefiltered", catalog.prefiltered());
        routingTrace.put("catalog_cached_stats_count", cachedStats);
        String prompt = AgenticIndexPrompts.routing(catalog.text(), request.getQuestion(), now, max);
        callLlm("route", prompt, ROUTE_MAX_TOKENS, ActionListener.wrap(completion -> {
            parseRouting(completion.content(), max);
            if (selected.isEmpty()) {
                warnings.add("routing selected no searchable index; used the best lexical matches");
                IndexCatalog.rank(profiles, request.getQuestion()).stream().limit(max).forEach(p -> select(p.getIndex(), "lexical match"));
            }
            listener.onResponse(null);
        }, listener::onFailure));
    }

    @SuppressWarnings("unchecked")
    void parseRouting(String content, int max) {
        Object items;
        try {
            items = LlmOutput.extractJsonObject(content).get("indices");
        } catch (IllegalArgumentException e) {
            warnings.add("routing output was not JSON");
            return;
        }
        if (!(items instanceof List)) {
            return;
        }
        int unknown = 0;
        int extra = 0;
        for (Object item : (List<Object>) items) {
            Object rawName = item instanceof Map ? ((Map<String, Object>) item).get("index") : item;
            if (rawName == null) {
                continue;
            }
            String name = String.valueOf(rawName).trim();
            if (!indices.containsKey(name)) {
                // Names are not echoed: the model may name indices the caller cannot read.
                unknown++;
            } else if (!selected.contains(name)) {
                if (selected.size() >= max) {
                    extra++;
                    continue;
                }
                Object reason = item instanceof Map ? ((Map<String, Object>) item).get("reason") : null;
                select(name, reason == null ? "" : PassageStore.truncate(String.valueOf(reason), 300));
            }
        }
        if (unknown > 0) {
            warnings.add("routing named " + unknown + " indices that are not among the resolved indices; ignored");
        }
        if (extra > 0) {
            warnings.add("routing selected more than " + max + " indices; extra ones dropped");
        }
    }

    private void addIndexInsight(ActionListener<Void> listener) {
        if (!request.isUseIndexInsight()) {
            listener.onResponse(null);
            return;
        }
        List<String> names = selected.stream().filter(name -> !indices.get(name).insightRequested).toList();
        names.forEach(name -> indices.get(name).insightRequested = true);
        BoundedFanOut.<String, String>run(names, PROFILE_CONCURRENCY, (name, insightListener) -> {
            MLIndexInsightGetRequest insightRequest = MLIndexInsightGetRequest
                .builder()
                .indexName(multi ? name : request.getIndex())
                .targetIndexInsight(MLIndexInsightType.FIELD_DESCRIPTION)
                .tenantId(request.getTenantId())
                .build();
            client
                .execute(
                    MLIndexInsightGetAction.INSTANCE,
                    insightRequest,
                    ActionListener
                        .wrap(insight -> insightListener.onResponse(insight.getIndexInsight().getContent()), insightListener::onFailure)
                );
        }, outcomes -> {
            for (int i = 0; i < names.size(); i++) {
                SearchableIndex index = indices.get(names.get(i));
                Outcome<String> outcome = outcomes.get(i);
                if (outcome.failure() == null) {
                    index.profile = index.profile.withIndexInsight(outcome.result());
                } else {
                    log.debug("Agentic index could not read index insight", outcome.failure());
                    warnings
                        .add(
                            prefix(index.name()) + "index insight unavailable (" + outcome.failure().getMessage() + "); planned without it"
                        );
                }
            }
            listener.onResponse(null);
        });
    }

    private SearchableIndex onlyIndex() {
        return indices.get(selected.iterator().next());
    }

    // Profiles of the selected indices, sharing one size budget, each with whether semantic search is available.
    private String renderSelectedProfiles() {
        int perIndex = Math.max(MIN_PROFILE_CHARS, PROFILES_MAX_CHARS / selected.size());
        StringBuilder sb = new StringBuilder();
        for (String name : selected) {
            SearchableIndex index = indices.get(name);
            sb.append(index.profile.render(perIndex));
            sb.append(index.semanticAvailable() ? "Semantic search: available\n" : "Semantic search: not available; use keyword\n");
            sb.append('\n');
        }
        return sb.toString();
    }

    // Catalog of the indices not selected yet, which a follow-up round may add.
    private String otherIndicesCatalog() {
        List<IndexProfile> others = indices.values().stream().filter(i -> !selected.contains(i.name())).map(i -> i.profile).toList();
        return others.isEmpty() ? "" : IndexCatalog.render(others, request.getQuestion(), OTHER_INDICES_MAX_CHARS).text();
    }

    private void plan(ActionListener<Map<String, Object>> listener) {
        String prompt;
        if (multi) {
            prompt = AgenticIndexPrompts.multiPlanner(renderSelectedProfiles(), request.getQuestion(), now, request.getMaxSubqueries());
        } else {
            SearchableIndex only = onlyIndex();
            prompt = AgenticIndexPrompts
                .planner(only.profile.render(), request.getQuestion(), now, request.getMaxSubqueries(), only.semanticAvailable());
        }
        callLlm("plan", prompt, PLAN_MAX_TOKENS, ActionListener.wrap(completion -> {
            List<SubQuery> subQueries = parseSubQueries(completion.content(), "subqueries", "plan", 0);
            if (subQueries.isEmpty()) {
                warnings.add("planner returned no usable subqueries; searched the question text");
                subQueries = new ArrayList<>();
                for (String name : selected) {
                    if (subQueries.size() >= request.getMaxSubqueries()) {
                        break;
                    }
                    SearchMode mode = indices.get(name).semanticAvailable() ? SearchMode.HYBRID : SearchMode.KEYWORD;
                    subQueries.add(new SubQuery(multi ? name : null, request.getQuestion(), mode, Map.of(), List.of()));
                }
            }
            executeRound(1, subQueries, listener);
        }, listener::onFailure));
    }

    /**
     * Parses planned subqueries. Each must name a selected index (implied when only one is selected); with
     * {@code addingRound} above 0, follow-ups may also add one profiled index that is not selected yet.
     */
    @SuppressWarnings("unchecked")
    List<SubQuery> parseSubQueries(String content, String key, String step, int addingRound) {
        List<SubQuery> result = new ArrayList<>();
        Object items;
        try {
            items = LlmOutput.extractJsonObject(content).get(key);
        } catch (IllegalArgumentException e) {
            warnings.add(step + " output was not JSON");
            return result;
        }
        if (!(items instanceof List)) {
            return result;
        }
        String added = null;
        int unknown = 0;
        int missing = 0;
        int extraIndices = 0;
        for (Object item : (List<Object>) items) {
            if (result.size() >= request.getMaxSubqueries()) {
                warnings.add(step + " proposed more than " + request.getMaxSubqueries() + " subqueries; extra ones dropped");
                break;
            }
            if (!(item instanceof Map)) {
                continue;
            }
            SearchableIndex target;
            Object rawIndex = ((Map<String, Object>) item).get("index");
            String name = rawIndex == null ? null : String.valueOf(rawIndex).trim();
            if (!multi || (name == null && selected.size() == 1)) {
                target = onlyIndex();
            } else if (name == null) {
                missing++;
                continue;
            } else if (selected.contains(name)) {
                target = indices.get(name);
            } else if (addingRound > 0 && indices.containsKey(name) && (added == null || added.equals(name))) {
                added = name;
                target = indices.get(name);
            } else if (addingRound > 0 && indices.containsKey(name)) {
                extraIndices++;
                continue;
            } else {
                unknown++;
                continue;
            }
            // A single-index subquery names no index, so its trace and prompts read as before multi-index support.
            SubQuery subQuery = SubQuery
                .fromPlan(item, multi ? target.name() : null, target.profile, target.semanticAvailable(), semanticUnavailableNote(target));
            if (subQuery != null && !alreadyExecuted(subQuery)) {
                result.add(subQuery);
            }
        }
        String addedIndex = added;
        if (addedIndex != null && result.stream().anyMatch(q -> addedIndex.equals(q.getIndex()))) {
            select(addedIndex, "added by the sufficiency check for follow-up round " + addingRound);
        }
        if (missing > 0) {
            warnings.add(step + " dropped " + missing + " subqueries that named no index");
        }
        if (unknown > 0) {
            warnings.add(step + " dropped " + unknown + " subqueries on indices that are not selected");
        }
        if (extraIndices > 0) {
            warnings.add(step + " dropped " + extraIndices + " subqueries on a second new index; a round may add one index");
        }
        return result;
    }

    private String semanticUnavailableNote(SearchableIndex index) {
        if (request.getEmbeddingModelId() == null) {
            return "semantic search not configured, used keyword";
        }
        return multi ? "index " + index.name() + " has no vector field, used keyword" : "no vector field found, used keyword";
    }

    private boolean alreadyExecuted(SubQuery candidate) {
        for (SubQuery done : executed) {
            if (Objects.equals(done.getIndex(), candidate.getIndex())
                && done.getText().equalsIgnoreCase(candidate.getText())
                && done.getFilters().equals(candidate.getFilters())) {
                return true;
            }
        }
        return false;
    }

    void executeRound(int round, List<SubQuery> subQueries, ActionListener<Map<String, Object>> listener) {
        executeRound(round, subQueries, done -> done.onResponse(null), listener);
    }

    /** Runs a round's subqueries while {@code alongside} runs; the sufficiency check starts once both are done. */
    private void executeRound(
        int round,
        List<SubQuery> subQueries,
        Consumer<ActionListener<Void>> alongside,
        ActionListener<Map<String, Object>> listener
    ) {
        if (stopped("search round " + round, listener)) {
            return;
        }
        long roundStart = System.nanoTime();
        executed.addAll(subQueries);
        AtomicReferenceArray<List<SearchHit>> hits = new AtomicReferenceArray<>(subQueries.size());
        AtomicReferenceArray<String> errors = new AtomicReferenceArray<>(subQueries.size());
        AtomicInteger pending = new AtomicInteger(subQueries.size());
        StepListener<Void> alongsideDone = new StepListener<>();
        alongside.accept(alongsideDone);
        Runnable searched = () -> {
            long searchTookMs = millisSince(roundStart);
            alongsideDone.whenComplete(v -> afterRound(round, subQueries, hits, errors, searchTookMs, listener), listener::onFailure);
        };
        if (subQueries.isEmpty()) {
            searched.run();
            return;
        }
        for (int i = 0; i < subQueries.size(); i++) {
            int idx = i;
            runSubQuery(subQueries.get(i), ActionListener.wrap(result -> {
                hits.set(idx, result);
                if (pending.decrementAndGet() == 0) {
                    searched.run();
                }
            }, e -> {
                // A model-planned subquery can be invalid (for example a bad date); report it and keep the others.
                log.debug("Agentic index subquery failed", e);
                errors.set(idx, e.getMessage());
                hits.set(idx, List.of());
                if (pending.decrementAndGet() == 0) {
                    searched.run();
                }
            }));
        }
    }

    private void afterRound(
        int round,
        List<SubQuery> subQueries,
        AtomicReferenceArray<List<SearchHit>> hits,
        AtomicReferenceArray<String> errors,
        long searchTookMs,
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
            roundTrace.put("search_took_ms", searchTookMs);
            rounds.add(roundTrace);
            judge(round, roundTrace, listener);
        } catch (Exception e) {
            listener.onFailure(e);
        }
    }

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
        String prompt;
        if (multi) {
            prompt = AgenticIndexPrompts
                .multiSufficiency(
                    renderSelectedProfiles(),
                    otherIndicesCatalog(),
                    request.getQuestion(),
                    now,
                    passages.render(passages.all()),
                    gson.toJson(previous),
                    request.getMaxSubqueries()
                );
        } else {
            SearchableIndex only = onlyIndex();
            prompt = AgenticIndexPrompts
                .sufficiency(
                    only.profile.render(),
                    request.getQuestion(),
                    now,
                    passages.render(passages.all()),
                    gson.toJson(previous),
                    request.getMaxSubqueries(),
                    only.semanticAvailable()
                );
        }
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
            List<SubQuery> followups = parseSubQueries(completion.content(), "followups", "sufficiency", round + 1);
            if (followups.isEmpty()) {
                warnings.add("evidence judged insufficient in round " + round + " but no new follow-up subqueries were proposed");
                finish(listener);
                return;
            }
            // An index the follow-ups added gets field statistics and index insight for the next sufficiency check; they
            // load while the round searches, which does not read them.
            executeRound(
                round + 1,
                followups,
                done -> addStats(ActionListener.wrap(v -> addIndexInsight(done), done::onFailure)),
                listener
            );
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
        List<Map<String, Object>> result = new ArrayList<>();
        if (passages == null) {
            return result;
        }
        if (lastJudgment.get("relevant") instanceof List) {
            ((List<String>) lastJudgment.get("relevant")).forEach(id -> result.add(passages.get(id)));
            passages.all().stream().filter(p -> (Integer) p.get("round") > judgedRound).forEach(result::add);
        } else {
            result.addAll(passages.all());
        }
        return result;
    }

    private void finish(ActionListener<Map<String, Object>> listener) {
        List<Map<String, Object>> selectedPassages = selectedPassages();
        if (request.getMode() == Mode.ASK && !selectedPassages.isEmpty() && !timedOut) {
            if (task.isCancelled()) {
                listener.onFailure(cancelled());
                return;
            }
            deadlinePassed("answer");
        }
        if (timedOut) {
            Map<String, Object> result = baseResult();
            if (request.getMode() == Mode.ASK) {
                result.put("answer", null);
            }
            result.put("passages", selectedPassages);
            result.put("trace", trace());
            listener.onResponse(result);
            return;
        }
        if (request.getMode() == Mode.RETRIEVE || selectedPassages.isEmpty()) {
            Map<String, Object> result = baseResult();
            if (request.getMode() == Mode.ASK) {
                result.put("answer", NO_EVIDENCE_ANSWER);
                result.put("citations", List.of());
            }
            result.put("passages", selectedPassages);
            result.put("trace", trace());
            listener.onResponse(result);
            return;
        }
        List<String> labels = new ArrayList<>();
        Map<String, String> labelToId = new LinkedHashMap<>();
        for (Map<String, Object> passage : selectedPassages) {
            String label = "P" + (labels.size() + 1);
            labels.add(label);
            labelToId.put(label, (String) passage.get("passage_id"));
        }
        callLlm(
            "answer",
            AgenticIndexPrompts.answer(request.getQuestion(), now, passages.render(selectedPassages, labels)),
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
                result.put("passages", selectedPassages);
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
        trace.put("resolved_index_count", resolvedCount);
        if (multi) {
            Map<String, Object> routing = new LinkedHashMap<>(routingTrace);
            routing.put("selected", selectedTrace);
            trace.put("routing", routing);
        }
        trace.put("rounds", rounds);
        if (!warnings.isEmpty()) {
            trace.put("warnings", warnings);
        }
        return trace;
    }

    private SearchableIndex indexOf(SubQuery subQuery) {
        return subQuery.getIndex() == null ? onlyIndex() : indices.get(subQuery.getIndex());
    }

    void runSubQuery(SubQuery subQuery, ActionListener<List<SearchHit>> listener) {
        ActionListener<List<SearchHit>> once = ActionListener.notifyOnce(listener);
        try {
            SearchableIndex index = indexOf(subQuery);
            int k = request.getK();
            boolean hasText = !subQuery.getText().isEmpty();
            if (!hasText || subQuery.getMode() == SearchMode.KEYWORD || !index.semanticAvailable()) {
                search(index, subQuery.keywordQuery(index.textFields), k, !hasText ? sortField(index, subQuery) : null, once);
                return;
            }
            QueryBuilder semantic = subQuery.semanticQuery(index.embeddingField, request.getEmbeddingModelId(), k);
            if (subQuery.getMode() == SearchMode.SEMANTIC) {
                search(index, semantic, k, null, once);
                return;
            }
            // Hybrid: keyword and semantic searches run in parallel and are fused when both return.
            AtomicReferenceArray<List<SearchHit>> results = new AtomicReferenceArray<>(2);
            AtomicInteger remaining = new AtomicInteger(2);
            List<QueryBuilder> queries = List.of(subQuery.keywordQuery(index.textFields), semantic);
            for (int i = 0; i < queries.size(); i++) {
                int idx = i;
                search(index, queries.get(i), k, null, ActionListener.wrap(result -> {
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
    private static String sortField(SearchableIndex index, SubQuery subQuery) {
        IndexProfile profile = index.profile;
        for (String field : subQuery.getFilters().keySet()) {
            if (profile.kindOf(field) == FieldKind.DATE && profile.hasDocValues(field)) {
                return field;
            }
        }
        return profile.fieldsOfKind(FieldKind.DATE).stream().filter(profile::hasDocValues).findFirst().orElse(null);
    }

    private void search(SearchableIndex index, QueryBuilder query, int k, String sortField, ActionListener<List<SearchHit>> listener) {
        SearchSourceBuilder source = new SearchSourceBuilder().query(query).size(k).trackTotalHits(false).timeout(remainingTime());
        source.fetchSource(sourceIncludes(index), index.excludedFields.toArray(new String[0]));
        if (sortField != null) {
            source.sort(sortField, SortOrder.DESC);
        }
        client.search(childSearch(index.resolved, source), ActionListener.wrap((SearchResponse response) -> {
            listener.onResponse(List.of(response.getHits().getHits()));
        }, listener::onFailure));
    }

    // Only fields a passage can use are fetched, so wide documents stay cheap: text, profiled and location fields.
    static String[] sourceIncludes(SearchableIndex index) {
        Set<String> includes = new LinkedHashSet<>(index.textFields);
        includes.addAll(index.profile.getProfiledFields());
        includes.add("title");
        includes.addAll(PassageStore.LOCATION_FIELDS);
        return includes.toArray(new String[0]);
    }

    // Searches the index's target; a target that spans several indices is narrowed to this one with an _index filter.
    private SearchRequest childSearch(ResolvedIndex index, SearchSourceBuilder source) {
        if (index.filterByIndex()) {
            BoolQueryBuilder narrowed = QueryBuilders.boolQuery().filter(QueryBuilders.termQuery("_index", index.name()));
            if (source.query() != null) {
                narrowed.must(source.query());
            }
            source.query(narrowed);
        }
        SearchRequest searchRequest = new SearchRequest(index.target()).source(source);
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
