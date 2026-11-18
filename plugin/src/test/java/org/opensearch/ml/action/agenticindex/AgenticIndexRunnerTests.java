/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.eq;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.verify;
import static org.mockito.Mockito.when;
import static org.opensearch.ml.action.agenticindex.PassageStoreTests.hit;

import java.nio.charset.StandardCharsets;
import java.time.Clock;
import java.time.Instant;
import java.time.ZoneOffset;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;

import org.apache.lucene.search.TotalHits;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.admin.indices.mapping.get.GetMappingsRequest;
import org.opensearch.action.admin.indices.mapping.get.GetMappingsResponse;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.tasks.TaskCancelledException;
import org.opensearch.core.tasks.TaskId;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.WrapperQueryBuilder;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.indexInsight.IndexInsight;
import org.opensearch.ml.common.indexInsight.MLIndexInsightType;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexRequest;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexRequest.Mode;
import org.opensearch.ml.common.transport.indexInsight.MLIndexInsightGetAction;
import org.opensearch.ml.common.transport.indexInsight.MLIndexInsightGetResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.search.SearchHit;
import org.opensearch.search.SearchHits;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.tasks.CancellableTask;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.transport.client.AdminClient;
import org.opensearch.transport.client.Client;
import org.opensearch.transport.client.IndicesAdminClient;

public class AgenticIndexRunnerTests extends OpenSearchTestCase {

    private static final Clock CLOCK = Clock.fixed(Instant.parse("2026-11-17T10:00:00Z"), ZoneOffset.UTC);

    private Client client;
    private IndicesAdminClient indices;
    private CancellableTask task;
    private IndexProfileCache profileCache;
    private final Deque<String> llmReplies = new ArrayDeque<>();
    private final List<String> prompts = new ArrayList<>();
    private final List<SearchRequest> searches = new ArrayList<>();
    private final List<SearchRequest> statsSearches = new ArrayList<>();
    private Function<String, List<SearchHit>> hitsForQuery = q -> List.of();
    private Exception statsFailure;
    private Exception llmFailure;
    private Map<String, Object> llmOutputOverride;
    // LLM call index (0-based) that sleeps before replying, to push a request past its deadline.
    private int slowCall = -1;
    private long slowCallMillis;
    // When set, search listeners are held here instead of being completed.
    private List<ActionListener<SearchResponse>> heldSearches;
    private boolean statsTimedOut;
    // Searches whose query contains this text sleep before replying, to push a request past its deadline.
    private String slowQuery;
    private long slowQueryMillis;
    private final List<TaskId> predictParents = new ArrayList<>();

    @Override
    public void setUp() throws Exception {
        super.setUp();
        client = mock(Client.class);
        AdminClient admin = mock(AdminClient.class);
        indices = mock(IndicesAdminClient.class);
        when(client.admin()).thenReturn(admin);
        when(admin.indices()).thenReturn(indices);
        mockMapping(indices, Map.of("properties", IndexProfileTests.PROPERTIES));
        task = mock(CancellableTask.class);
        profileCache = new IndexProfileCache(index -> "uuid/1", IndexProfileCache.TTL);

        doAnswer(invocation -> {
            SearchRequest request = invocation.getArgument(0);
            ActionListener<SearchResponse> listener = invocation.getArgument(1);
            assertNotNull("searches carry the request deadline", request.source().timeout());
            SearchResponse response = mock(SearchResponse.class);
            if (request.source().size() == 0) {
                statsSearches.add(request);
                if (statsFailure != null) {
                    listener.onFailure(statsFailure);
                } else {
                    when(response.getHits()).thenReturn(SearchHits.empty());
                    when(response.isTimedOut()).thenReturn(statsTimedOut);
                    listener.onResponse(response);
                }
                return null;
            }
            searches.add(request);
            if (heldSearches != null) {
                heldSearches.add(listener);
                return null;
            }
            String query = describe(request.source().query());
            if (slowQuery != null && query.contains(slowQuery)) {
                Thread.sleep(slowQueryMillis);
            }
            List<SearchHit> hits = hitsForQuery.apply(query);
            if (hits == null) {
                listener.onFailure(new IllegalArgumentException("failed to parse date field"));
                return null;
            }
            when(response.getHits())
                .thenReturn(new SearchHits(hits.toArray(new SearchHit[0]), new TotalHits(hits.size(), TotalHits.Relation.EQUAL_TO), 1f));
            listener.onResponse(response);
            return null;
        }).when(client).search(any(SearchRequest.class), any());

        doAnswer(invocation -> {
            MLPredictionTaskRequest request = invocation.getArgument(1);
            ActionListener<MLTaskResponse> listener = invocation.getArgument(2);
            prompts.add(((RemoteInferenceInputDataSet) request.getMlInput().getInputDataset()).getParameters().get("messages"));
            predictParents.add(request.getParentTask());
            if (prompts.size() - 1 == slowCall) {
                Thread.sleep(slowCallMillis);
            }
            if (llmFailure != null) {
                listener.onFailure(llmFailure);
                return null;
            }
            if (llmOutputOverride != null) {
                listener.onResponse(MLTaskResponse.builder().output(LlmOutputTests.output(llmOutputOverride)).build());
                return null;
            }
            String reply = llmReplies.poll();
            assertNotNull("unexpected LLM call", reply);
            Map<String, Object> data = Map
                .of(
                    "choices",
                    List.of(Map.of("message", Map.of("content", reply))),
                    "usage",
                    Map.of("prompt_tokens", 100, "completion_tokens", 10)
                );
            listener.onResponse(MLTaskResponse.builder().output(LlmOutputTests.output(data)).build());
            return null;
        }).when(client).execute(eq(MLPredictionTaskAction.INSTANCE), any(), any());
    }

    // Wrapper (neural) queries print base64 bytes, so decode them for routing.
    private static String describe(QueryBuilder query) {
        if (query instanceof WrapperQueryBuilder wrapper) {
            return new String(wrapper.source(), StandardCharsets.UTF_8);
        }
        return query.toString();
    }

    private void mockMapping(IndicesAdminClient indices, Map<String, Object> mapping) {
        mockMappings(indices, Map.of("docs", new MappingMetadata("docs", mapping)));
    }

    private void mockMappings(IndicesAdminClient indices, Map<String, MappingMetadata> mappings) {
        doAnswer(invocation -> {
            ActionListener<GetMappingsResponse> listener = invocation.getArgument(1);
            GetMappingsResponse response = mock(GetMappingsResponse.class);
            when(response.getMappings()).thenReturn(mappings);
            listener.onResponse(response);
            return null;
        }).when(indices).getMappings(any(GetMappingsRequest.class), any());
    }

    private AgenticIndexRunner runner(MLAgenticIndexRequest request) {
        return runner(request, TimeValue.timeValueMinutes(1));
    }

    private AgenticIndexRunner runner(MLAgenticIndexRequest request, TimeValue timeout) {
        return runner(request, timeout, "user|roles");
    }

    private AgenticIndexRunner runner(MLAgenticIndexRequest request, TimeValue timeout, String caller) {
        return new AgenticIndexRunner(request, client, CLOCK, profileCache, caller, timeout, task, "node-1");
    }

    private MLAgenticIndexRequest request(Mode mode, int maxRounds, boolean semantic) {
        MLAgenticIndexRequest.MLAgenticIndexRequestBuilder builder = MLAgenticIndexRequest
            .builder()
            .index("docs")
            .mode(mode)
            .question("Which plans include SSO and what changed in pricing?")
            .llmModelId("llm")
            .maxRounds(maxRounds);
        if (semantic) {
            builder.embeddingModelId("emb").embeddingField("embedding");
        }
        return builder.build();
    }

    private Map<String, Object> run(MLAgenticIndexRequest request) {
        return run(runner(request));
    }

    private Map<String, Object> run(AgenticIndexRunner runner) {
        AtomicReference<Map<String, Object>> result = new AtomicReference<>();
        AtomicReference<Exception> failure = new AtomicReference<>();
        runner.run(ActionListener.wrap(result::set, failure::set));
        if (failure.get() != null) {
            throw new AssertionError(failure.get());
        }
        assertNotNull("runner did not complete", result.get());
        return result.get();
    }

    private Exception runExpectingFailure(MLAgenticIndexRequest request) {
        AtomicReference<Exception> failure = new AtomicReference<>();
        runner(request).run(ActionListener.wrap(r -> fail("expected failure"), failure::set));
        assertNotNull("runner did not fail", failure.get());
        return failure.get();
    }

    private void routeHits() {
        hitsForQuery = q -> {
            if (q.contains("\"price change\"")) {
                return List
                    .of(hit("c", "{\"title\":\"Pricing update\",\"body\":\"Pro rose to $24\",\"source\":\"pricing.pdf\",\"page\":4}"));
            }
            if (q.contains("\"sso\"")) {
                return List.of(hit("a", "{\"title\":\"Security\",\"body\":\"SSO in Pro\",\"source\":\"pricing.pdf\",\"page\":3}"));
            }
            if (q.contains("\"pricing\"")) {
                return List.of(hit("b", "{\"body\":\"Free plan\"}"), hit("a", "{\"body\":\"SSO in Pro\"}"));
            }
            return List.of();
        };
    }

    @SuppressWarnings("unchecked")
    public void testAskWithFollowUpRound() {
        routeHits();
        llmReplies
            .add(
                "<think>plan</think>{\"subqueries\":[{\"query\":\"sso\",\"search\":\"keyword\",\"filters\":{\"product\":\"A\"}},"
                    + "{\"query\":\"pricing\",\"search\":\"keyword\"}]}"
            );
        llmReplies
            .add(
                "{\"sufficient\":false,\"relevant\":[\"P1\"],\"missing\":\"price change\",\"followups\":["
                    + "{\"query\":\"price change\",\"search\":\"keyword\"},{\"query\":\"sso\",\"filters\":{\"product\":\"A\"}}]}"
            );
        llmReplies.add("{\"sufficient\":true,\"relevant\":[\"P3\",\"P1\",\"P9\"]}");
        // The answer step relabels its passages by position: P1 = passage P3, P2 = passage P1.
        llmReplies.add("SSO is in Pro [P2]. Pro rose to $24 [P1]. Also [P7].");

        Map<String, Object> result = run(request(Mode.ASK, 3, false));

        assertEquals("ask", result.get("mode"));
        assertEquals(true, result.get("sufficient"));
        assertEquals("SSO is in Pro [P1]. Pro rose to $24 [P3]. Also.", result.get("answer"));
        List<Map<String, Object>> citations = (List<Map<String, Object>>) result.get("citations");
        assertEquals(List.of("a", "c"), citations.stream().map(c -> c.get("doc_id")).toList());
        assertEquals(4, citations.get(1).get("page"));
        List<Map<String, Object>> passages = (List<Map<String, Object>>) result.get("passages");
        assertEquals(List.of("P3", "P1"), passages.stream().map(p -> p.get("passage_id")).toList());

        Map<String, Object> trace = (Map<String, Object>) result.get("trace");
        assertEquals(4, trace.get("llm_call_count"));
        assertEquals(400L, trace.get("prompt_tokens"));
        assertEquals(40L, trace.get("completion_tokens"));
        List<Map<String, Object>> rounds = (List<Map<String, Object>>) trace.get("rounds");
        assertEquals(2, rounds.size());
        // The duplicate follow-up (sso + product filter) is not re-run.
        assertEquals(1, ((List<?>) rounds.get(1).get("subqueries")).size());
        assertEquals(List.of("answer cited passages that were not provided [P7]; markers removed"), trace.get("warnings"));
        assertEquals(3, searches.size());
        assertTrue(prompts.get(3).contains("[P1] doc_id=c"));
        assertFalse("answer prompt only carries relevant passages", prompts.get(3).contains("doc_id=b"));
    }

    @SuppressWarnings("unchecked")
    public void testRetrieveStopsAtMaxRounds() {
        routeHits();
        llmReplies.add("{\"subqueries\":[{\"query\":\"pricing\"}]}");
        llmReplies.add("{\"sufficient\":false,\"relevant\":[\"P2\"],\"missing\":\"x\",\"followups\":[{\"query\":\"price change\"}]}");

        Map<String, Object> result = run(request(Mode.RETRIEVE, 1, false));

        assertEquals("retrieve", result.get("mode"));
        assertEquals(false, result.get("sufficient"));
        assertEquals("x", result.get("missing"));
        assertFalse(result.containsKey("answer"));
        List<Map<String, Object>> passages = (List<Map<String, Object>>) result.get("passages");
        assertEquals(List.of("a"), passages.stream().map(p -> p.get("doc_id")).toList());
        assertEquals(2, ((Map<String, Object>) result.get("trace")).get("llm_call_count"));
        assertTrue(llmReplies.isEmpty());
    }

    @SuppressWarnings("unchecked")
    public void testPlannerFallbackAndFailedSubQuery() {
        hitsForQuery = q -> q.contains("\"bad\"") ? null : List.of(hit("a", "{\"body\":\"SSO in Pro\"}"));
        llmReplies.add("I cannot produce JSON");
        llmReplies.add("not json either");
        llmReplies.add("SSO is in Pro [P1].");

        Map<String, Object> result = run(request(Mode.ASK, 2, false));
        Map<String, Object> trace = (Map<String, Object>) result.get("trace");
        List<String> warnings = (List<String>) trace.get("warnings");
        assertTrue(warnings.contains("plan output was not JSON"));
        assertTrue(warnings.contains("planner returned no usable subqueries; searched the question text"));
        assertTrue(warnings.contains("sufficiency output was not JSON in round 1; treated as sufficient"));
        assertEquals(1, ((List<?>) result.get("citations")).size());

        prompts.clear();
        llmReplies.add("{\"subqueries\":[{\"query\":\"bad\"},{\"query\":\"good\"}]}");
        llmReplies.add("{\"sufficient\":true,\"relevant\":[\"P1\"]}");
        llmReplies.add("Answer [P1].");
        result = run(request(Mode.ASK, 2, false));
        List<Map<String, Object>> rounds = (List<Map<String, Object>>) ((Map<String, Object>) result.get("trace")).get("rounds");
        List<Map<String, Object>> subQueries = (List<Map<String, Object>>) rounds.get(0).get("subqueries");
        assertEquals("failed to parse date field", subQueries.get(0).get("error"));
        assertEquals(List.of("P1"), subQueries.get(1).get("passages"));
    }

    public void testNoEvidenceSkipsAnswerCall() {
        llmReplies.add("{\"subqueries\":[{\"query\":\"sla credit\"}]}");

        Map<String, Object> result = run(request(Mode.ASK, 1, false));

        assertEquals(AgenticIndexRunner.NO_EVIDENCE_ANSWER, result.get("answer"));
        assertEquals(List.of(), result.get("citations"));
        assertEquals("no passages retrieved", result.get("missing"));
        assertTrue(llmReplies.isEmpty());
        assertEquals(1, prompts.size());
    }

    public void testHybridSubQueryRunsKeywordAndSemantic() {
        hitsForQuery = q -> q.contains("neural") ? List.of(hit("s", "{\"body\":\"semantic\"}")) : List.of(hit("k", "{\"body\":\"kw\"}"));
        llmReplies.add("{\"subqueries\":[{\"query\":\"single sign-on\",\"search\":\"hybrid\"},{\"filters\":{\"product\":\"A\"}}]}");
        llmReplies.add("{\"sufficient\":true}");
        llmReplies.add("Answer [P1] [P2].");

        Map<String, Object> result = run(request(Mode.ASK, 1, true));

        // hybrid = keyword + semantic search; filter-only subquery = one keyword search sorted by date
        assertEquals(3, searches.size());
        assertTrue(searches.get(2).source().sorts().get(0).toString().contains("published_at"));
        assertFalse(prompts.get(0).contains("Semantic search is not available"));
        assertEquals(2, ((List<?>) result.get("citations")).size());
    }

    public void testEmptyRelevantListMeansNoEvidence() {
        routeHits();
        llmReplies.add("{\"subqueries\":[{\"query\":\"pricing\"}]}");
        llmReplies.add("{\"sufficient\":false,\"relevant\":[],\"missing\":\"SLA terms\",\"followups\":[]}");

        Map<String, Object> result = run(request(Mode.ASK, 2, false));

        assertEquals(AgenticIndexRunner.NO_EVIDENCE_ANSWER, result.get("answer"));
        assertEquals(List.of(), result.get("passages"));
        assertEquals("SLA terms", result.get("missing"));
        assertTrue(llmReplies.isEmpty());
    }

    public void testReciprocalRankFusion() {
        SearchHit a = hit("a", "{}");
        SearchHit b = hit("b", "{}");
        SearchHit c = hit("c", "{}");
        List<SearchHit> fused = AgenticIndexRunner.reciprocalRankFusion(List.of(a, b), List.of(b, c), 2);
        assertEquals(List.of("b", "a"), fused.stream().map(SearchHit::getId).toList());
    }

    public void testFailsWithoutTextFields() {
        IndicesAdminClient indices = client.admin().indices();
        mockMapping(indices, Map.of("properties", Map.of("product", Map.of("type", "keyword"))));
        Exception e = runExpectingFailure(request(Mode.ASK, 1, false));
        assertEquals("Index [docs] has no text fields to search", e.getMessage());
    }

    public void testUsesIndexInsightWhenRequested() {
        doAnswer(invocation -> {
            ActionListener<MLIndexInsightGetResponse> listener = invocation.getArgument(2);
            IndexInsight insight = IndexInsight
                .builder()
                .index("docs")
                .taskType(MLIndexInsightType.FIELD_DESCRIPTION)
                .content("product: product line name")
                .build();
            listener.onResponse(MLIndexInsightGetResponse.builder().indexInsight(insight).build());
            return null;
        }).when(client).execute(eq(MLIndexInsightGetAction.INSTANCE), any(), any());
        llmReplies.add("{\"subqueries\":[]}");
        MLAgenticIndexRequest request = MLAgenticIndexRequest
            .builder()
            .index("docs")
            .mode(Mode.RETRIEVE)
            .question("q")
            .llmModelId("llm")
            .maxRounds(1)
            .useIndexInsight(true)
            .build();

        run(request);

        assertTrue(prompts.get(0).contains("Index insight:"));
        assertTrue(prompts.get(0).contains("product: product line name"));
    }

    public void testIndexInsightNotCalledByDefault() {
        llmReplies.add("{\"subqueries\":[]}");
        run(request(Mode.RETRIEVE, 1, false));
        verify(client, never()).execute(eq(MLIndexInsightGetAction.INSTANCE), any(), any());
    }

    public void testLlmCallFailureFailsRequest() {
        llmFailure = new OpenSearchStatusException("model not found", RestStatus.NOT_FOUND);
        Exception e = runExpectingFailure(request(Mode.ASK, 1, false));
        assertSame(llmFailure, e);
    }

    public void testMalformedModelOutputIsBadRequest() {
        llmOutputOverride = Map.of("output", Map.of("message", "not a chat completion"));
        OpenSearchStatusException e = (OpenSearchStatusException) runExpectingFailure(request(Mode.ASK, 1, false));
        assertEquals(RestStatus.BAD_REQUEST, e.status());
        assertEquals(
            "Model llm did not return a chat completion. Use a connector that returns choices[0].message.content.",
            e.getMessage()
        );
    }

    @SuppressWarnings("unchecked")
    public void testStatsSearchFailureIsWarning() {
        statsFailure = new IllegalArgumentException("Can't load fielddata on [code]");
        llmReplies.add("{\"subqueries\":[{\"query\":\"sso\"}]}");

        Map<String, Object> result = run(request(Mode.RETRIEVE, 1, false));

        List<String> warnings = (List<String>) ((Map<String, Object>) result.get("trace")).get("warnings");
        assertTrue(
            warnings.toString(),
            warnings.contains("index field statistics unavailable (Can't load fielddata on [code]); planned without field values")
        );
        assertEquals(1, prompts.size());
    }

    @SuppressWarnings("unchecked")
    public void testObjectPathTextFields() {
        mockMapping(
            indices,
            Map
                .of(
                    "properties",
                    Map
                        .of(
                            "content",
                            Map.of("properties", Map.of("body", Map.of("type", "text"), "lang", Map.of("type", "keyword"))),
                            "source",
                            Map.of("type", "keyword")
                        )
                )
        );
        hitsForQuery = q -> List.of(hit("a", "{\"content\":{\"body\":\"SSO in Pro\",\"lang\":\"en\"},\"source\":\"a.pdf\"}"));
        llmReplies.add("{\"subqueries\":[{\"query\":\"sso\"}]}");
        llmReplies.add("{\"sufficient\":true}");

        Map<String, Object> result = run(request(Mode.RETRIEVE, 1, false));

        Map<String, Object> passage = ((List<Map<String, Object>>) result.get("passages")).get(0);
        assertEquals("SSO in Pro", passage.get("text"));
        assertEquals(Map.of("source", "a.pdf"), passage.get("metadata"));
        assertTrue(searches.get(0).source().query().toString().contains("content.body"));
    }

    public void testRejectsTargetWithSeveralIndices() {
        Map<String, Object> mapping = Map.of("properties", IndexProfileTests.PROPERTIES);
        mockMappings(indices, Map.of("docs-1", new MappingMetadata("docs-1", mapping), "docs-2", new MappingMetadata("docs-2", mapping)));

        OpenSearchStatusException e = (OpenSearchStatusException) runExpectingFailure(request(Mode.ASK, 1, false));

        assertEquals(RestStatus.BAD_REQUEST, e.status());
        assertTrue(e.getMessage(), e.getMessage().startsWith("[docs] resolves to 2 indices [docs-1, docs-2]."));
        assertTrue(prompts.isEmpty());
    }

    public void testDeadlineBeforePlanningReturnsEmptyPartialResult() {
        Map<String, Object> result = run(runner(request(Mode.ASK, 2, false), TimeValue.timeValueNanos(1)));

        assertEquals(true, result.get("timed_out"));
        assertEquals(List.of(), result.get("passages"));
        assertNull(result.get("answer"));
        assertTrue(prompts.isEmpty());
    }

    @SuppressWarnings("unchecked")
    public void testDeadlineBeforeAnswerReturnsRetrievedPassages() {
        routeHits();
        llmReplies.add("{\"subqueries\":[{\"query\":\"sso\"}]}");
        llmReplies.add("{\"sufficient\":true}");
        slowCall = 1;
        slowCallMillis = 2500;

        Map<String, Object> result = run(runner(request(Mode.ASK, 2, false), TimeValue.timeValueSeconds(2)));

        assertEquals(true, result.get("timed_out"));
        assertNull(result.get("answer"));
        assertTrue(result.containsKey("answer"));
        assertEquals(List.of("a"), ((List<Map<String, Object>>) result.get("passages")).stream().map(p -> p.get("doc_id")).toList());
        Map<String, Object> trace = (Map<String, Object>) result.get("trace");
        assertEquals(2, trace.get("llm_call_count"));
        assertEquals(List.of("request deadline reached before answer; returned partial results"), trace.get("warnings"));
    }

    public void testCancelledTaskStopsBeforePlanning() {
        when(task.isCancelled()).thenReturn(true);
        when(task.getReasonCancelled()).thenReturn("by user");
        Exception e = runExpectingFailure(request(Mode.ASK, 1, false));
        assertTrue(e instanceof TaskCancelledException);
        assertEquals("The agentic index request was cancelled: by user", e.getMessage());
        assertTrue(prompts.isEmpty());
    }

    public void testIndexInsightFailureIsWarning() {
        doAnswer(invocation -> {
            ActionListener<MLIndexInsightGetResponse> listener = invocation.getArgument(2);
            listener.onFailure(new IllegalStateException("index insight not enabled"));
            return null;
        }).when(client).execute(eq(MLIndexInsightGetAction.INSTANCE), any(), any());
        llmReplies.add("{\"subqueries\":[]}");
        MLAgenticIndexRequest request = MLAgenticIndexRequest
            .builder()
            .index("docs")
            .mode(Mode.RETRIEVE)
            .question("q")
            .llmModelId("llm")
            .maxRounds(1)
            .useIndexInsight(true)
            .build();

        Map<String, Object> result = run(request);

        @SuppressWarnings("unchecked")
        List<String> warnings = (List<String>) ((Map<String, Object>) result.get("trace")).get("warnings");
        assertTrue(warnings.toString(), warnings.contains("index insight unavailable (index insight not enabled); planned without it"));
        assertFalse(prompts.get(0).contains("Index insight:"));
    }

    public void testEmptyRoundCompletes() {
        AgenticIndexRunner runner = runner(request(Mode.RETRIEVE, 1, false));
        runner.buildProfile(ActionListener.wrap(p -> {}, e -> fail(e.getMessage())));
        AtomicReference<Map<String, Object>> result = new AtomicReference<>();
        runner.executeRound(1, List.of(), ActionListener.wrap(result::set, e -> fail(e.getMessage())));
        assertNotNull(result.get());
        assertEquals(List.of(), result.get().get("passages"));
    }

    public void testProfileIsCachedPerCaller() {
        llmReplies.add("{\"subqueries\":[]}");
        llmReplies.add("{\"subqueries\":[]}");
        llmReplies.add("{\"subqueries\":[]}");
        run(request(Mode.RETRIEVE, 1, false));
        run(request(Mode.RETRIEVE, 1, false));
        assertEquals(1, statsSearches.size());
        run(runner(request(Mode.RETRIEVE, 1, false), TimeValue.timeValueMinutes(1), "other|roles"));
        assertEquals(2, statsSearches.size());
    }

    public void testProfileIsCachedPerAlias() {
        for (int i = 0; i < 3; i++) {
            llmReplies.add("{\"subqueries\":[]}");
        }
        // Both filtered aliases resolve to the concrete index "docs" but can see different documents.
        run(aliasRequest("alias-a"));
        run(aliasRequest("alias-b"));
        assertEquals(2, statsSearches.size());
        run(aliasRequest("alias-a"));
        assertEquals(2, statsSearches.size());
        assertEquals(List.of("alias-a", "alias-b"), statsSearches.stream().map(r -> r.indices()[0]).toList());
    }

    private static MLAgenticIndexRequest aliasRequest(String alias) {
        return MLAgenticIndexRequest.builder().index(alias).mode(Mode.RETRIEVE).question("q").llmModelId("llm").maxRounds(1).build();
    }

    @SuppressWarnings("unchecked")
    public void testIncompleteStatsAreNotCached() {
        statsTimedOut = true;
        llmReplies.add("{\"subqueries\":[]}");
        llmReplies.add("{\"subqueries\":[]}");

        Map<String, Object> result = run(request(Mode.RETRIEVE, 1, false));
        run(request(Mode.RETRIEVE, 1, false));

        assertEquals(2, statsSearches.size());
        List<String> warnings = (List<String>) ((Map<String, Object>) result.get("trace")).get("warnings");
        assertTrue(
            warnings.toString(),
            warnings.contains("index field statistics are incomplete (timed out or failed shards); planned with partial field values")
        );
    }

    public void testChildRequestsCarryParentTask() {
        when(task.getId()).thenReturn(42L);
        routeHits();
        llmReplies.add("{\"subqueries\":[{\"query\":\"sso\"}]}");
        llmReplies.add("{\"sufficient\":true}");

        run(request(Mode.RETRIEVE, 1, false));

        TaskId parent = new TaskId("node-1", 42L);
        assertEquals(parent, statsSearches.get(0).getParentTask());
        assertEquals(parent, searches.get(0).getParentTask());
        assertEquals(2, predictParents.size());
        predictParents.forEach(p -> assertEquals(parent, p));
    }

    public void testSearchesFetchOnlyUsableFields() {
        routeHits();
        llmReplies.add("{\"subqueries\":[{\"query\":\"sso\"}]}");
        llmReplies.add("{\"sufficient\":true}");

        run(request(Mode.RETRIEVE, 1, true));

        FetchSourceContext fetch = searches.get(0).source().fetchSource();
        List<String> includes = List.of(fetch.includes());
        assertTrue(includes.toString(), includes.containsAll(List.of("title", "body", "product", "published_at", "meta.archived")));
        assertTrue(includes.toString(), includes.containsAll(PassageStore.LOCATION_FIELDS));
        assertFalse(includes.toString(), includes.contains("embedding"));
        assertTrue(List.of(fetch.excludes()).contains("embedding"));
    }

    @SuppressWarnings("unchecked")
    public void testDeadlineAfterFollowUpRoundKeepsJudgedAndNewPassages() {
        routeHits();
        llmReplies.add("{\"subqueries\":[{\"query\":\"pricing\"}]}");
        llmReplies.add("{\"sufficient\":false,\"relevant\":[\"P2\"],\"missing\":\"change\",\"followups\":[{\"query\":\"price change\"}]}");
        slowQuery = "\"price change\"";
        slowQueryMillis = 2500;

        Map<String, Object> result = run(runner(request(Mode.ASK, 3, false), TimeValue.timeValueSeconds(2)));

        assertEquals(true, result.get("timed_out"));
        assertTrue(result.containsKey("answer"));
        assertNull(result.get("answer"));
        // P2 was judged relevant in round 1; P3 comes from round 2, which the deadline left unjudged. P1 was judged irrelevant.
        List<Map<String, Object>> passages = (List<Map<String, Object>>) result.get("passages");
        assertEquals(List.of("P2", "P3"), passages.stream().map(p -> p.get("passage_id")).toList());
        Map<String, Object> trace = (Map<String, Object>) result.get("trace");
        assertEquals(
            List.of("request deadline reached before sufficiency check of round 2; returned partial results"),
            trace.get("warnings")
        );
        assertEquals(2, trace.get("llm_call_count"));
    }

    public void testHybridSearchesRunInParallel() {
        AgenticIndexRunner runner = runner(request(Mode.RETRIEVE, 1, true));
        runner.buildProfile(ActionListener.wrap(p -> {}, e -> fail(e.getMessage())));
        heldSearches = new ArrayList<>();
        AtomicReference<List<SearchHit>> fused = new AtomicReference<>();
        SubQuery subQuery = new SubQuery("sso", SubQuery.SearchMode.HYBRID, Map.of(), List.of());

        runner.runSubQuery(subQuery, ActionListener.wrap(fused::set, e -> fail(e.getMessage())));

        // Both searches are issued before either returns.
        assertEquals(2, heldSearches.size());
        for (int i = 1; i >= 0; i--) {
            SearchResponse response = mock(SearchResponse.class);
            SearchHit hit = hit(i == 0 ? "k" : "s", "{\"body\":\"x\"}");
            when(response.getHits()).thenReturn(new SearchHits(new SearchHit[] { hit }, new TotalHits(1, TotalHits.Relation.EQUAL_TO), 1f));
            heldSearches.get(i).onResponse(response);
        }
        assertEquals(List.of("k", "s"), fused.get().stream().map(SearchHit::getId).toList());
    }

    public void testHybridSearchFailureFailsSubQueryOnce() {
        AgenticIndexRunner runner = runner(request(Mode.RETRIEVE, 1, true));
        runner.buildProfile(ActionListener.wrap(p -> {}, e -> fail(e.getMessage())));
        heldSearches = new ArrayList<>();
        List<Exception> failures = new ArrayList<>();
        runner
            .runSubQuery(
                new SubQuery("sso", SubQuery.SearchMode.HYBRID, Map.of(), List.of()),
                ActionListener.wrap(r -> fail("expected failure"), failures::add)
            );
        heldSearches.get(0).onFailure(new IllegalArgumentException("bad"));
        heldSearches.get(1).onFailure(new IllegalArgumentException("bad again"));
        assertEquals(1, failures.size());
    }
}
