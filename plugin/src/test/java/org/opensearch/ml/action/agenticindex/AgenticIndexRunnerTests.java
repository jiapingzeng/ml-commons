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
import java.util.ArrayList;
import java.util.Collections;
import java.util.Deque;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedDeque;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiFunction;
import java.util.function.BooleanSupplier;
import java.util.function.Function;

import org.apache.lucene.search.TotalHits;
import org.opensearch.OpenSearchSecurityException;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.Version;
import org.opensearch.action.admin.indices.mapping.get.GetMappingsRequest;
import org.opensearch.action.admin.indices.mapping.get.GetMappingsResponse;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.search.ShardSearchFailure;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.tasks.TaskCancelledException;
import org.opensearch.core.tasks.TaskId;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.WrapperQueryBuilder;
import org.opensearch.ml.action.agenticindex.IndexResolver.ResolvedIndex;
import org.opensearch.ml.common.dataset.remote.RemoteInferenceInputDataSet;
import org.opensearch.ml.common.indexInsight.IndexInsight;
import org.opensearch.ml.common.indexInsight.MLIndexInsightType;
import org.opensearch.ml.common.transport.MLTaskResponse;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexRequest;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexRequest.Mode;
import org.opensearch.ml.common.transport.indexInsight.MLIndexInsightGetAction;
import org.opensearch.ml.common.transport.indexInsight.MLIndexInsightGetRequest;
import org.opensearch.ml.common.transport.indexInsight.MLIndexInsightGetResponse;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskRequest;
import org.opensearch.search.SearchHit;
import org.opensearch.search.SearchHits;
import org.opensearch.search.fetch.subphase.FetchSourceContext;
import org.opensearch.search.internal.InternalSearchResponse;
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
    private final Deque<String> llmReplies = new ConcurrentLinkedDeque<>();
    private final List<String> prompts = Collections.synchronizedList(new ArrayList<>());
    private final List<SearchRequest> searches = Collections.synchronizedList(new ArrayList<>());
    private final List<SearchRequest> statsSearches = Collections.synchronizedList(new ArrayList<>());
    // Searches complete on another thread, as they do on a cluster; set to false to complete them on the calling thread.
    private boolean asyncSearches = true;
    private ExecutorService searchExecutor;
    private final Set<String> searchCompletionThreads = ConcurrentHashMap.newKeySet();
    private ClusterState clusterState = ClusterState.EMPTY_STATE;
    private Function<String, List<SearchHit>> hitsForQuery = q -> List.of();
    // When set, routes searches by target and query instead of hitsForQuery.
    private BiFunction<SearchRequest, String, List<SearchHit>> hitsForSearch;
    // When set, the runner gets these indices instead of resolving the target through the mapping mock.
    private List<ResolvedIndex> resolvedIndices;
    // When set, resolving the target fails with this exception.
    private Exception resolveFailure;
    private final List<String> statsFailingIndices = new ArrayList<>();
    private Exception statsFailure;
    private Exception llmFailure;
    private Map<String, Object> llmOutputOverride;
    // LLM call index (0-based) that sleeps before replying, to push a request past its deadline.
    private int slowCall = -1;
    private long slowCallMillis;
    // When set, search listeners are held here instead of being completed.
    private List<ActionListener<SearchResponse>> heldSearches;
    private boolean statsTimedOut;
    // Statistics search of this index is held in heldStats instead of being completed.
    private String holdStatsOf;
    private final AtomicReference<Runnable> heldStats = new AtomicReference<>();
    // Searches whose query contains this text sleep before replying, to push a request past its deadline.
    private String slowQuery;
    private long slowQueryMillis;
    private final List<TaskId> predictParents = Collections.synchronizedList(new ArrayList<>());

    @Override
    public void setUp() throws Exception {
        super.setUp();
        searchExecutor = Executors.newFixedThreadPool(4);
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
            if (request.source().size() == 0) {
                statsSearches.add(request);
                String index = request.indices()[0];
                if (index.equals(holdStatsOf)) {
                    heldStats.set(() -> listener.onResponse(response(List.of(), false)));
                    return null;
                }
                complete(() -> {
                    if (statsFailingIndices.contains(index)) {
                        listener.onFailure(new IllegalArgumentException("stats failed"));
                    } else if (statsFailure != null) {
                        listener.onFailure(statsFailure);
                    } else {
                        listener.onResponse(response(List.of(), statsTimedOut));
                    }
                });
                return null;
            }
            searches.add(request);
            if (heldSearches != null) {
                heldSearches.add(listener);
                return null;
            }
            String query = describe(request.source().query());
            complete(() -> {
                if (slowQuery != null && query.contains(slowQuery)) {
                    sleep(slowQueryMillis);
                }
                List<SearchHit> hits = hitsForSearch != null ? hitsForSearch.apply(request, query) : hitsForQuery.apply(query);
                if (hits == null) {
                    listener.onFailure(new IllegalArgumentException("failed to parse date field"));
                } else {
                    listener.onResponse(response(hits, false));
                }
            });
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

    @Override
    public void tearDown() throws Exception {
        searchExecutor.shutdown();
        assertTrue(searchExecutor.awaitTermination(10, TimeUnit.SECONDS));
        super.tearDown();
    }

    private void complete(Runnable completion) {
        Runnable recorded = () -> {
            searchCompletionThreads.add(Thread.currentThread().getName());
            completion.run();
        };
        if (asyncSearches) {
            searchExecutor.execute(recorded);
        } else {
            recorded.run();
        }
    }

    private static void sleep(long millis) {
        try {
            Thread.sleep(millis);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException(e);
        }
    }

    private static SearchResponse response(List<SearchHit> hits, boolean timedOut) {
        SearchHits searchHits = new SearchHits(hits.toArray(new SearchHit[0]), new TotalHits(hits.size(), TotalHits.Relation.EQUAL_TO), 1f);
        InternalSearchResponse sections = new InternalSearchResponse(searchHits, null, null, null, timedOut, null, 1);
        return new SearchResponse(sections, null, 1, 1, 0, 1, ShardSearchFailure.EMPTY_ARRAY, SearchResponse.Clusters.EMPTY);
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
        IndexResolver resolver = new IndexResolver(
            client,
            () -> clusterState,
            new IndexNameExpressionResolver(new ThreadContext(Settings.EMPTY))
        ) {
            @Override
            public void resolve(
                String expression,
                int max,
                TaskId parent,
                BooleanSupplier cancelled,
                BooleanSupplier expired,
                ActionListener<List<ResolvedIndex>> l
            ) {
                if (resolveFailure != null) {
                    l.onFailure(resolveFailure);
                } else if (resolvedIndices != null) {
                    l.onResponse(resolvedIndices);
                } else {
                    super.resolve(expression, max, parent, cancelled, expired, l);
                }
            }
        };
        return new AgenticIndexRunner(request, client, CLOCK, profileCache, resolver, 500, caller, timeout, task, "node-1");
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
        await(runner, result, failure);
        if (failure.get() != null) {
            throw new AssertionError(failure.get());
        }
        assertNotNull("runner did not complete", result.get());
        return result.get();
    }

    private Exception runExpectingFailure(MLAgenticIndexRequest request) {
        AtomicReference<Map<String, Object>> result = new AtomicReference<>();
        AtomicReference<Exception> failure = new AtomicReference<>();
        await(runner(request), result, failure);
        assertNull("expected failure", result.get());
        assertNotNull("runner did not fail", failure.get());
        return failure.get();
    }

    // Runs to completion; fails the test when the runner never completes its listener.
    private static void await(AgenticIndexRunner runner, AtomicReference<Map<String, Object>> result, AtomicReference<Exception> failure) {
        CountDownLatch done = new CountDownLatch(1);
        runner.run(ActionListener.wrap(r -> {
            result.set(r);
            done.countDown();
        }, e -> {
            failure.set(e);
            done.countDown();
        }));
        try {
            assertTrue("runner did not complete its listener", done.await(30, TimeUnit.SECONDS));
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new AssertionError(e);
        }
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
        // A single-index request names no index on its subqueries, in the trace or the sufficiency prompt.
        rounds.forEach(r -> ((List<Map<String, Object>>) r.get("subqueries")).forEach(q -> assertFalse(q.containsKey("index"))));
        assertFalse(prompts.get(1), prompts.get(1).contains("\\\"index\\\""));
        assertFalse(trace.containsKey("routing"));
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

    public void testPatternResolvingToSeveralIndicesIsMultiIndex() {
        Map<String, Object> mapping = Map.of("properties", IndexProfileTests.PROPERTIES);
        mockMappings(indices, Map.of("docs-1", new MappingMetadata("docs-1", mapping), "docs-2", new MappingMetadata("docs-2", mapping)));
        llmReplies.add("{\"subqueries\":[]}");

        Map<String, Object> result = run(request(Mode.RETRIEVE, 1, false));

        Map<String, Object> trace = trace(result);
        assertEquals(2, trace.get("resolved_index_count"));
        assertTrue(prompts.get(0).contains("Index: docs-1"));
        assertTrue(prompts.get(0).contains("Index: docs-2"));
        assertEquals(List.of("docs-1", "docs-2"), statsSearches.stream().map(r -> r.indices()[0]).sorted().toList());
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
        runner.resolveAndProfile(ActionListener.wrap(p -> {}, e -> fail(e.getMessage())));
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
        runner.resolveAndProfile(ActionListener.wrap(p -> {}, e -> fail(e.getMessage())));
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
        runner.resolveAndProfile(ActionListener.wrap(p -> {}, e -> fail(e.getMessage())));
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

    // ---- multi-index -------------------------------------------------------

    private static final Map<String, Object> PRICING = Map
        .of("title", Map.of("type", "text"), "body", Map.of("type", "text"), "plan", Map.of("type", "keyword"));
    private static final Map<String, Object> TICKETS = Map
        .of(
            "summary",
            Map.of("type", "text"),
            "severity",
            Map.of("type", "keyword"),
            "opened_at",
            Map.of("type", "date"),
            "vec",
            Map.of("type", "knn_vector", "dimension", 8)
        );
    private static final Map<String, Object> NOTES = Map.of("note", Map.of("type", "text"));

    private static ResolvedIndex resolved(String name, Map<String, Object> properties) {
        return new ResolvedIndex(name, name, false, properties, null);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> trace(Map<String, Object> result) {
        return (Map<String, Object>) result.get("trace");
    }

    @SuppressWarnings("unchecked")
    private static List<String> warnings(Map<String, Object> result) {
        return (List<String>) trace(result).getOrDefault("warnings", List.of());
    }

    private MLAgenticIndexRequest multiRequest(Mode mode, int maxRounds, int maxSelected) {
        return MLAgenticIndexRequest
            .builder()
            .index("docs-*")
            .mode(mode)
            .question("What does the Pro plan cost and was there an SSO outage?")
            .llmModelId("llm")
            .embeddingModelId("emb")
            .maxRounds(maxRounds)
            .maxIndicesSelected(maxSelected)
            .build();
    }

    private void routeHitsByIndex() {
        hitsForSearch = (request, query) -> switch (request.indices()[0]) {
            case "docs-a" -> List.of(hit("docs-a", "a1", "{\"body\":\"Pro costs $20\",\"plan\":\"Pro\",\"source\":\"pricing.pdf\"}"));
            case "docs-b" -> List.of(hit("docs-b", "b1", "{\"summary\":\"SSO outage on 2026-10-02\",\"severity\":\"high\"}"));
            case "docs-c" -> List.of(hit("docs-c", "c1", "{\"note\":\"team offsite\"}"));
            default -> List.of();
        };
    }

    @SuppressWarnings("unchecked")
    public void testMultiIndexRoutesPlansAndCitesPerIndex() {
        resolvedIndices = List.of(resolved("docs-a", PRICING), resolved("docs-b", TICKETS), resolved("docs-c", NOTES));
        routeHitsByIndex();
        llmReplies
            .add(
                "{\"indices\":[{\"index\":\"docs-b\",\"reason\":\"incident tickets\"},{\"index\":\"other-team\",\"reason\":\"x\"},"
                    + "{\"index\":\"docs-a\",\"reason\":\"pricing\"},{\"index\":\"docs-c\"}]}"
            );
        llmReplies
            .add(
                "{\"subqueries\":[{\"index\":\"docs-a\",\"query\":\"pro price\",\"search\":\"hybrid\","
                    + "\"filters\":{\"plan\":\"Pro\",\"severity\":\"high\"}},{\"index\":\"docs-b\",\"query\":\"sso outage\",\"search\":\"hybrid\"},"
                    + "{\"index\":\"docs-c\",\"query\":\"offsite\"},{\"query\":\"no index\"}]}"
            );
        llmReplies.add("{\"sufficient\":true,\"relevant\":[\"P1\",\"P2\"]}");
        llmReplies.add("Pro costs $20 [P1]. There was an SSO outage [P2].");

        Map<String, Object> result = run(runner(multiRequest(Mode.ASK, 2, 2)));

        String routingPrompt = prompts.get(0);
        assertTrue(routingPrompt, routingPrompt.contains("- docs-a: fields ") && routingPrompt.contains("plan(keyword)"));
        assertTrue(routingPrompt, routingPrompt.contains("- docs-b: fields"));
        assertTrue(routingPrompt, routingPrompt.contains("- docs-c: fields note(text)"));
        String planPrompt = prompts.get(1);
        assertTrue(planPrompt.contains("Index: docs-a") && planPrompt.contains("Index: docs-b"));
        assertFalse(planPrompt.contains("Index: docs-c"));
        assertTrue(planPrompt.contains("Semantic search: not available; use keyword"));

        Map<String, Object> trace = trace(result);
        assertEquals(3, trace.get("resolved_index_count"));
        Map<String, Object> routing = (Map<String, Object>) trace.get("routing");
        assertEquals(true, routing.get("routed"));
        assertEquals(3, routing.get("catalog_index_count"));
        assertEquals(false, routing.get("catalog_prefiltered"));
        assertTrue((Integer) routing.get("catalog_chars") > 0);
        assertEquals(
            List.of(Map.of("index", "docs-b", "reason", "incident tickets"), Map.of("index", "docs-a", "reason", "pricing")),
            routing.get("selected")
        );
        List<String> warnings = warnings(result);
        assertTrue(warnings.toString(), warnings.contains("routing named 1 indices that are not among the resolved indices; ignored"));
        assertTrue(warnings.toString(), warnings.contains("routing selected more than 2 indices; extra ones dropped"));
        assertTrue(warnings.toString(), warnings.contains("plan dropped 1 subqueries that named no index"));
        assertTrue(warnings.toString(), warnings.contains("plan dropped 1 subqueries on indices that are not selected"));
        assertFalse("unknown index names are not echoed", trace.toString().contains("other-team"));

        List<Map<String, Object>> subQueries = (List<Map<String, Object>>) ((List<Map<String, Object>>) trace.get("rounds"))
            .get(0)
            .get("subqueries");
        assertEquals(List.of("docs-a", "docs-b"), subQueries.stream().map(q -> q.get("index")).toList());
        // docs-a has no vector field, so its hybrid subquery ran as keyword; severity is not a docs-a field.
        assertEquals("keyword", subQueries.get(0).get("search"));
        assertEquals(Map.of("plan", "Pro"), subQueries.get(0).get("filters"));
        assertTrue(subQueries.get(0).get("warnings").toString().contains("index docs-a has no vector field, used keyword"));
        assertTrue(subQueries.get(0).get("warnings").toString().contains("dropped filter on non-filterable field severity"));
        assertEquals("hybrid", subQueries.get(1).get("search"));

        // docs-a: one keyword search; docs-b: keyword plus neural on the detected vector field.
        assertEquals(List.of("docs-a", "docs-b", "docs-b"), searches.stream().map(r -> r.indices()[0]).toList());
        assertTrue(describe(searches.get(2).source().query()).contains("\"vec\""));
        assertTrue(searches.get(0).source().query().toString().contains("body"));
        assertFalse(searches.get(0).source().query().toString().contains("summary"));

        assertTrue(prompts.get(3).contains("[P1] index=docs-a doc_id=a1"));
        List<Map<String, Object>> citations = (List<Map<String, Object>>) result.get("citations");
        assertEquals(List.of("docs-a", "docs-b"), citations.stream().map(c -> c.get("index")).toList());
        List<Map<String, Object>> passages = (List<Map<String, Object>>) result.get("passages");
        assertEquals(List.of("docs-a", "docs-b"), passages.stream().map(p -> p.get("index")).toList());
        assertEquals("Pro costs $20", passages.get(0).get("text"));
        assertEquals("SSO outage on 2026-10-02", passages.get(1).get("text"));
    }

    @SuppressWarnings("unchecked")
    public void testSufficiencyMayAddOneIndexPerRound() {
        resolvedIndices = List.of(resolved("docs-a", PRICING), resolved("docs-b", TICKETS), resolved("docs-c", NOTES));
        routeHitsByIndex();
        llmReplies.add("{\"indices\":[{\"index\":\"docs-a\",\"reason\":\"pricing\"}]}");
        llmReplies.add("{\"subqueries\":[{\"index\":\"docs-a\",\"query\":\"pro price\"}]}");
        llmReplies
            .add(
                "{\"sufficient\":false,\"missing\":\"outage\",\"followups\":[{\"index\":\"docs-b\",\"query\":\"sso outage\"},"
                    + "{\"index\":\"docs-c\",\"query\":\"offsite\"},{\"index\":\"docs-b\",\"query\":\"sso\"}]}"
            );
        llmReplies.add("{\"sufficient\":true}");

        Map<String, Object> result = run(runner(multiRequest(Mode.RETRIEVE, 3, 1)));

        assertTrue(prompts.get(2).contains("Other indices (a follow-up subquery may search at most one of them"));
        assertTrue(prompts.get(2).contains("- docs-b: fields") && prompts.get(2).contains("- docs-c: fields"));
        assertFalse(prompts.get(2).contains("- docs-a: fields"));
        // After docs-b was added, only docs-c is left to add.
        assertTrue(prompts.get(3).contains("Index: docs-b"));
        assertFalse(prompts.get(3).contains("- docs-b: fields"));

        Map<String, Object> trace = trace(result);
        List<Map<String, Object>> rounds = (List<Map<String, Object>>) trace.get("rounds");
        List<Map<String, Object>> followups = (List<Map<String, Object>>) rounds.get(1).get("subqueries");
        assertEquals(List.of("docs-b", "docs-b"), followups.stream().map(q -> q.get("index")).toList());
        List<Map<String, Object>> selected = (List<Map<String, Object>>) ((Map<String, Object>) trace.get("routing")).get("selected");
        assertEquals(List.of("docs-a", "docs-b"), selected.stream().map(e -> e.get("index")).toList());
        assertEquals("added by the sufficiency check for follow-up round 2", selected.get(1).get("reason"));
        assertTrue(
            warnings(result).toString(),
            warnings(result).contains("sufficiency dropped 1 subqueries on a second new index; a round may add one index")
        );
        assertEquals(
            List.of("docs-a", "docs-b"),
            ((List<Map<String, Object>>) result.get("passages")).stream().map(p -> p.get("index")).toList()
        );
        // Only the routed index and the one a follow-up added get a statistics search.
        assertEquals(List.of("docs-a", "docs-b"), statsSearches.stream().map(r -> r.indices()[0]).toList());
    }

    @SuppressWarnings("unchecked")
    public void testRoutingCatalogShowsCachedValuesAndDescription() {
        resolvedIndices = List
            .of(
                new ResolvedIndex("docs-a", "docs-a", false, PRICING, "Plan prices\nby tier"),
                resolved("docs-b", TICKETS),
                resolved("docs-c", NOTES)
            );
        MLAgenticIndexRequest request = multiRequest(Mode.RETRIEVE, 1, 1);
        profileCache
            .put(
                profileCache.key("user|roles", "docs-b", "docs-b", request.getTextFields()),
                IndexCatalogTests.withKeywordValues(IndexProfile.fromMappingProperties("docs-b", TICKETS), "severity", List.of("high"))
            );
        // Another caller's cached values are not shown.
        profileCache
            .put(
                profileCache.key("other|roles", "docs-c", "docs-c", request.getTextFields()),
                IndexCatalogTests.withKeywordValues(IndexProfile.fromMappingProperties("docs-c", NOTES), "note", List.of("hidden"))
            );
        llmReplies.add("{\"indices\":[{\"index\":\"docs-a\",\"reason\":\"pricing\"}]}");
        llmReplies.add("{\"subqueries\":[]}");

        Map<String, Object> result = run(runner(request));

        String routingPrompt = prompts.get(0);
        assertTrue(routingPrompt, routingPrompt.contains("- docs-a: description Plan prices by tier; fields "));
        assertTrue(routingPrompt, routingPrompt.contains("- docs-b: fields ") && routingPrompt.contains("; values severity=high"));
        assertFalse(routingPrompt, routingPrompt.contains("hidden"));
        assertEquals(1, ((Map<String, Object>) trace(result).get("routing")).get("catalog_cached_stats_count"));
        // Routing reads the cache only; the statistics search runs for the selected index.
        assertEquals(List.of("docs-a"), statsSearches.stream().map(r -> r.indices()[0]).toList());
    }

    public void testFollowUpIndexGetsStatsAndInsightWhileItsRoundSearches() {
        resolvedIndices = List.of(resolved("docs-a", PRICING), resolved("docs-b", TICKETS), resolved("docs-c", NOTES));
        List<String> insightIndices = Collections.synchronizedList(new ArrayList<>());
        doAnswer(invocation -> {
            MLIndexInsightGetRequest insightRequest = invocation.getArgument(1);
            ActionListener<MLIndexInsightGetResponse> listener = invocation.getArgument(2);
            insightIndices.add(insightRequest.getIndexName());
            IndexInsight insight = IndexInsight
                .builder()
                .index(insightRequest.getIndexName())
                .taskType(MLIndexInsightType.FIELD_DESCRIPTION)
                .content("insight for " + insightRequest.getIndexName())
                .build();
            listener.onResponse(MLIndexInsightGetResponse.builder().indexInsight(insight).build());
            return null;
        }).when(client).execute(eq(MLIndexInsightGetAction.INSTANCE), any(), any());
        holdStatsOf = "docs-b";
        AtomicBoolean statsPendingAtSearch = new AtomicBoolean();
        hitsForSearch = (request, query) -> {
            if (request.indices()[0].equals("docs-b")) {
                Runnable stats = heldStats.getAndSet(null);
                if (stats != null) {
                    statsPendingAtSearch.set(true);
                    stats.run();
                }
                return List.of(hit("docs-b", "b1", "{\"summary\":\"SSO outage\"}"));
            }
            return List.of(hit("docs-a", "a1", "{\"body\":\"Pro costs $20\"}"));
        };
        llmReplies.add("{\"indices\":[{\"index\":\"docs-a\",\"reason\":\"pricing\"}]}");
        llmReplies.add("{\"subqueries\":[{\"index\":\"docs-a\",\"query\":\"pro price\"}]}");
        llmReplies.add("{\"sufficient\":false,\"missing\":\"outage\",\"followups\":[{\"index\":\"docs-b\",\"query\":\"sso\"}]}");
        llmReplies.add("{\"sufficient\":true}");
        MLAgenticIndexRequest request = MLAgenticIndexRequest
            .builder()
            .index("docs-*")
            .mode(Mode.RETRIEVE)
            .question("What does the Pro plan cost and was there an SSO outage?")
            .llmModelId("llm")
            .maxRounds(2)
            .maxIndicesSelected(1)
            .useIndexInsight(true)
            .build();

        run(runner(request));

        assertTrue("the round searched while the added index's statistics were pending", statsPendingAtSearch.get());
        assertEquals(List.of("docs-a", "docs-b"), insightIndices);
        assertTrue(prompts.get(3), prompts.get(3).contains("insight for docs-b") && prompts.get(3).contains("insight for docs-a"));
        assertNull(heldStats.get());
    }

    @SuppressWarnings("unchecked")
    public void testFailedIndexIsSkippedWithWarning() {
        Map<String, Object> keywordOnly = Map.of("code", Map.of("type", "keyword"));
        resolvedIndices = List.of(resolved("docs-a", PRICING), resolved("docs-b", keywordOnly), resolved("docs-c", NOTES));
        statsFailingIndices.add("docs-c");
        llmReplies.add("{\"subqueries\":[{\"index\":\"docs-c\",\"query\":\"offsite\"}]}");

        Map<String, Object> result = run(runner(multiRequest(Mode.RETRIEVE, 1, 5)));

        Map<String, Object> trace = trace(result);
        assertEquals(3, trace.get("resolved_index_count"));
        Map<String, Object> routing = (Map<String, Object>) trace.get("routing");
        assertEquals(2, routing.get("profiled_index_count"));
        assertEquals(false, routing.get("routed"));
        assertEquals(
            List.of("docs-a", "docs-c"),
            ((List<Map<String, Object>>) routing.get("selected")).stream().map(e -> e.get("index")).toList()
        );
        assertEquals(
            List
                .of(
                    "index [docs-b] skipped: Index [docs-b] has no text fields to search",
                    "index [docs-c]: index field statistics unavailable (stats failed); planned without field values",
                    "index [docs-a]: no vector field; searched with keyword only",
                    "index [docs-c]: no vector field; searched with keyword only"
                )
                .stream()
                .sorted()
                .toList(),
            warnings(result).stream().sorted().toList()
        );
        // No passages in the last round, so the sufficiency check is skipped.
        assertEquals(1, trace.get("llm_call_count"));
    }

    public void testFailsWhenNoIndexCanBeSearched() {
        Map<String, Object> keywordOnly = Map.of("code", Map.of("type", "keyword"));
        resolvedIndices = List.of(resolved("docs-a", keywordOnly), resolved("docs-b", keywordOnly));
        AtomicReference<Exception> failure = new AtomicReference<>();
        runner(multiRequest(Mode.ASK, 1, 5)).run(ActionListener.wrap(r -> fail("expected failure"), failure::set));
        OpenSearchStatusException e = (OpenSearchStatusException) failure.get();
        assertEquals(RestStatus.BAD_REQUEST, e.status());
        assertEquals(
            "None of the 2 indices that [docs-*] resolves to can be searched: index [docs-a] skipped: Index [docs-a] has no text "
                + "fields to search; index [docs-b] skipped: Index [docs-b] has no text fields to search",
            e.getMessage()
        );
        assertTrue(prompts.isEmpty());
    }

    public void testRequestedTextFieldsApplyPerIndex() {
        resolvedIndices = List.of(resolved("docs-a", PRICING), resolved("docs-b", TICKETS));
        llmReplies.add("{\"subqueries\":[{\"index\":\"docs-a\",\"query\":\"pro\"}]}");
        MLAgenticIndexRequest request = MLAgenticIndexRequest
            .builder()
            .index("docs-*")
            .mode(Mode.RETRIEVE)
            .question("q")
            .llmModelId("llm")
            .textFields(List.of("body", "missing"))
            .maxRounds(1)
            .build();

        Map<String, Object> result = run(request);

        assertTrue(
            warnings(result).contains("index [docs-b] skipped: Index [docs-b] has none of the requested text fields [body, missing]")
        );
        assertTrue(searches.get(0).source().query().toString().contains("\"body^1.0\""));
        assertFalse(searches.get(0).source().query().toString().contains("missing"));
    }

    @SuppressWarnings("unchecked")
    public void testRoutingFallsBackToLexicalMatches() {
        resolvedIndices = List.of(resolved("billing-invoices", PRICING), resolved("hr-notes", NOTES), resolved("support-tickets", TICKETS));
        llmReplies.add("no idea");
        llmReplies.add("{\"subqueries\":[]}");
        MLAgenticIndexRequest request = MLAgenticIndexRequest
            .builder()
            .index("*")
            .mode(Mode.RETRIEVE)
            .question("Which invoices were paid late?")
            .llmModelId("llm")
            .maxRounds(1)
            .maxIndicesSelected(1)
            .build();

        Map<String, Object> result = run(request);

        assertTrue(warnings(result).contains("routing output was not JSON"));
        assertTrue(warnings(result).contains("routing selected no searchable index; used the best lexical matches"));
        Map<String, Object> routing = (Map<String, Object>) trace(result).get("routing");
        assertEquals(List.of(Map.of("index", "billing-invoices", "reason", "lexical match")), routing.get("selected"));
        // The planner fallback searches the question text in the selected index.
        assertEquals(List.of("billing-invoices"), searches.stream().map(r -> r.indices()[0]).toList());
    }

    @SuppressWarnings("unchecked")
    public void testAliasTargetIsNarrowedToIndex() {
        resolvedIndices = List
            .of(
                new ResolvedIndex("docs-a", "pricing-alias", true, PRICING, null),
                new ResolvedIndex("docs-b", "docs-b", false, TICKETS, null)
            );
        routeHitsByIndex();
        hitsForSearch = (request, query) -> request.indices()[0].equals("pricing-alias")
            ? List.of(hit("docs-a", "a1", "{\"body\":\"Pro costs $20\"}"))
            : List.of();
        llmReplies.add("{\"subqueries\":[{\"index\":\"docs-a\",\"query\":\"pro\"}]}");
        llmReplies.add("{\"sufficient\":true}");

        Map<String, Object> result = run(runner(multiRequest(Mode.RETRIEVE, 1, 5)));

        SearchRequest statsSearch = statsSearches.stream().filter(r -> r.indices()[0].equals("pricing-alias")).findFirst().orElseThrow();
        assertTrue(statsSearch.source().query().toString().contains("\"_index\""));
        assertEquals("pricing-alias", searches.get(0).indices()[0]);
        String query = searches.get(0).source().query().toString();
        assertTrue(query, query.contains("\"_index\"") && query.contains("\"docs-a\""));
        assertEquals("docs-a", ((List<Map<String, Object>>) result.get("passages")).get(0).get("index"));
    }

    public void testMultiIndexDeadlineBeforeRouting() {
        resolvedIndices = List.of(resolved("docs-a", PRICING), resolved("docs-b", TICKETS), resolved("docs-c", NOTES));

        Map<String, Object> result = run(runner(multiRequest(Mode.ASK, 2, 1), TimeValue.timeValueNanos(1)));

        assertEquals(true, result.get("timed_out"));
        assertNull(result.get("answer"));
        assertTrue(
            warnings(result).toString(),
            warnings(result).contains("request deadline reached before routing; returned partial results")
        );
        assertEquals(3, trace(result).get("resolved_index_count"));
        assertTrue("profiles past the deadline skip the statistics search", statsSearches.isEmpty());
        assertTrue(prompts.isEmpty());
    }

    public void testMultiIndexCancelledBeforeRouting() {
        resolvedIndices = List.of(resolved("docs-a", PRICING), resolved("docs-b", TICKETS));
        when(task.isCancelled()).thenReturn(true);
        AtomicReference<Exception> failure = new AtomicReference<>();
        runner(multiRequest(Mode.ASK, 1, 1)).run(ActionListener.wrap(r -> fail("expected failure"), failure::set));
        assertTrue(failure.get() instanceof TaskCancelledException);
        assertTrue(prompts.isEmpty());
        assertTrue(statsSearches.isEmpty());
    }

    public void testProfilesAreCachedPerIndex() {
        resolvedIndices = List.of(resolved("docs-a", PRICING), resolved("docs-b", TICKETS));
        llmReplies.add("{\"subqueries\":[]}");
        llmReplies.add("{\"subqueries\":[]}");
        run(runner(multiRequest(Mode.RETRIEVE, 1, 5)));
        run(runner(multiRequest(Mode.RETRIEVE, 1, 5)));
        assertEquals(2, statsSearches.size());
    }

    public void testResolutionTimeoutReturnsEmptyPartialResult() {
        resolveFailure = new IndexResolver.TimedOutException("timed out");
        Map<String, Object> result = run(runner(multiRequest(Mode.ASK, 1, 1), TimeValue.timeValueNanos(1)));
        assertEquals(true, result.get("timed_out"));
        assertNull(result.get("answer"));
        assertEquals(List.of(), result.get("passages"));
        assertEquals(List.of("request deadline reached before the target was resolved; returned partial results"), warnings(result));
        assertEquals(0, trace(result).get("resolved_index_count"));
        assertTrue(prompts.isEmpty());
    }

    public void testIndicesWithoutTextFieldsFailOrSkipWhenSearchesCompleteOnAnotherThread() {
        Map<String, Object> keywordOnly = Map.of("code", Map.of("type", "keyword"));
        resolvedIndices = List.of(resolved("docs-a", PRICING), resolved("docs-b", keywordOnly));
        hitsForQuery = q -> List.of(hit("docs-a", "a1", "{\"body\":\"Pro costs $20\"}"));
        llmReplies.add("{\"subqueries\":[{\"index\":\"docs-a\",\"query\":\"pro\"}]}");
        llmReplies.add("{\"sufficient\":true}");

        Map<String, Object> result = run(runner(multiRequest(Mode.RETRIEVE, 1, 5)));

        assertTrue(warnings(result).contains("index [docs-b] skipped: Index [docs-b] has no text fields to search"));
        assertFalse("searches completed on another thread", searchCompletionThreads.contains(Thread.currentThread().getName()));
        assertFalse(searchCompletionThreads.isEmpty());

        resolvedIndices = List.of(resolved("docs", keywordOnly));
        assertEquals("Index [docs] has no text fields to search", runExpectingFailure(request(Mode.ASK, 1, false)).getMessage());
    }

    @SuppressWarnings("unchecked")
    public void testPartialAccessResolutionKeepsUnreadableIndexOutOfRouting() {
        clusterState = clusterState("docs-a", "docs-b", "docs-c", "docs-secret");
        Map<String, Map<String, Object>> properties = Map.of("docs-a", PRICING, "docs-b", TICKETS, "docs-c", NOTES);
        // The caller may read docs-a, docs-b and docs-c by name, but neither the pattern (it matches docs-secret) nor docs-secret.
        doAnswer(invocation -> {
            GetMappingsRequest mappingRequest = invocation.getArgument(0);
            ActionListener<GetMappingsResponse> listener = invocation.getArgument(1);
            String[] names = mappingRequest.indices();
            if (names.length != 1 || !properties.containsKey(names[0])) {
                listener
                    .onFailure(new OpenSearchSecurityException("no permissions for [indices:admin/mappings/get]", RestStatus.FORBIDDEN));
                return null;
            }
            GetMappingsResponse response = mock(GetMappingsResponse.class);
            when(response.getMappings())
                .thenReturn(Map.of(names[0], new MappingMetadata(names[0], Map.of("properties", properties.get(names[0])))));
            listener.onResponse(response);
            return null;
        }).when(indices).getMappings(any(GetMappingsRequest.class), any());
        routeHitsByIndex();
        llmReplies.add("{\"indices\":[{\"index\":\"docs-secret\",\"reason\":\"x\"},{\"index\":\"docs-a\",\"reason\":\"pricing\"}]}");
        llmReplies.add("{\"subqueries\":[{\"index\":\"docs-secret\",\"query\":\"falcon\"},{\"index\":\"docs-a\",\"query\":\"pro\"}]}");
        llmReplies.add("{\"sufficient\":false,\"missing\":\"x\",\"followups\":[{\"index\":\"docs-secret\",\"query\":\"merger\"}]}");

        Map<String, Object> result = run(runner(multiRequest(Mode.RETRIEVE, 2, 1)));

        Map<String, Object> trace = trace(result);
        assertEquals(3, trace.get("resolved_index_count"));
        assertEquals(3, ((Map<String, Object>) trace.get("routing")).get("catalog_index_count"));
        assertTrue(prompts.get(0), prompts.get(0).contains("- docs-b: fields") && prompts.get(0).contains("- docs-c: fields"));
        prompts.forEach(p -> assertFalse(p, p.contains("docs-secret")));
        assertFalse(result.toString(), result.toString().contains("docs-secret"));
        List<String> targets = new ArrayList<>();
        searches.forEach(r -> targets.add(r.indices()[0]));
        statsSearches.forEach(r -> targets.add(r.indices()[0]));
        assertEquals(List.of("docs-a", "docs-a"), targets);
        assertEquals(
            List.of("docs-a"),
            ((List<Map<String, Object>>) result.get("passages")).stream().map(p -> p.get("index")).distinct().toList()
        );
        assertTrue(warnings(result).contains("routing named 1 indices that are not among the resolved indices; ignored"));
        assertTrue(warnings(result).contains("plan dropped 1 subqueries on indices that are not selected"));
        assertTrue(warnings(result).contains("sufficiency dropped 1 subqueries on indices that are not selected"));
    }

    private static ClusterState clusterState(String... names) {
        Metadata.Builder metadata = Metadata.builder();
        for (String name : names) {
            metadata
                .put(
                    IndexMetadata
                        .builder(name)
                        .settings(
                            Settings
                                .builder()
                                .put(IndexMetadata.SETTING_VERSION_CREATED, Version.CURRENT)
                                .put(IndexMetadata.SETTING_NUMBER_OF_SHARDS, 1)
                                .put(IndexMetadata.SETTING_NUMBER_OF_REPLICAS, 0)
                        )
                );
        }
        return ClusterState.builder(new ClusterName("test")).metadata(metadata).build();
    }
}
