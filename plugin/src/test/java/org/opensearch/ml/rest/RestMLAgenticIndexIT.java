/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_AGENTIC_INDEX_ENABLED;
import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_AGENTIC_INDEX_MAX_INDICES;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.ml.common.MLTaskState;
import org.opensearch.ml.utils.TestHelper;

import com.google.common.collect.ImmutableMap;

/**
 * Covers the agentic index REST routes against a real cluster: the feature gate, request validation, target
 * resolution, and full multi-index ask and retrieve runs against a mock LLM.
 */
public class RestMLAgenticIndexIT extends MLCommonsRestTestCase {

    private String index;
    private final List<String> extraIndices = new ArrayList<>();
    private AgenticIndexMockLlm mockLlm;

    @Before
    public void setUpIndex() throws IOException {
        index = "agentic_index_" + randomAlphaOfLength(8).toLowerCase(Locale.ROOT);
        request(
            "PUT",
            "/" + index,
            "{\"mappings\":{\"properties\":{\"title\":{\"type\":\"text\"},\"plan\":{\"type\":\"keyword\"},"
                + "\"code\":{\"type\":\"keyword\",\"doc_values\":false},\"published_at\":{\"type\":\"date\"}}}}"
        );
        request(
            "PUT",
            "/" + index + "/_doc/1?refresh=true",
            "{\"title\":\"SSO is in Pro\",\"plan\":\"Pro\",\"published_at\":\"2026-10-01\"}"
        );
    }

    @After
    public void tearDownIndex() throws IOException {
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), null);
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_MAX_INDICES.getKey(), null);
        deleteIndexWithAdminClient(index);
        for (String extra : extraIndices) {
            deleteIndexWithAdminClient(extra);
        }
        if (mockLlm != null) {
            mockLlm.close();
            updateClusterSettings("plugins.ml_commons.trusted_connector_endpoints_regex", null);
            updateClusterSettings("plugins.ml_commons.connector.private_ip_enabled", null);
        }
    }

    public void testDisabledByDefault() {
        ResponseException e = expectThrows(
            ResponseException.class,
            () -> request("POST", "/_plugins/_ml/agentic_index/" + index + "/_ask", "{\"question\":\"q\",\"llm_model_id\":\"m\"}")
        );
        assertEquals(403, e.getResponse().getStatusLine().getStatusCode());
        assertTrue(e.getMessage(), e.getMessage().contains("agentic index APIs are not enabled"));
    }

    @SuppressWarnings("unchecked")
    public void testDeadlineReturnsPartialResultAfterProfiling() throws IOException {
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), true);
        Response response = request(
            "POST",
            "/_plugins/_ml/agentic_index/" + index + "/_ask",
            "{\"question\":\"Which plans include SSO?\",\"llm_model_id\":\"no-such-model\",\"timeout\":\"1nanos\"}"
        );
        Map<String, Object> result = parseResponseToMap(response);
        assertEquals(true, result.get("timed_out"));
        assertEquals(List.of(), result.get("passages"));
        Map<String, Object> trace = (Map<String, Object>) result.get("trace");
        // The statistics search skipped the keyword field without doc values, so it produced no warning.
        assertEquals(List.of("request deadline reached before planning; returned partial results"), trace.get("warnings"));
    }

    @SuppressWarnings("unchecked")
    public void testIndicesBodyFieldResolvesSeveralIndices() throws IOException {
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), true);
        String prefix = createMultiIndices();
        Response response = request(
            "POST",
            "/_plugins/_ml/agentic_index/_retrieve",
            "{\"question\":\"q\",\"llm_model_id\":\"m\",\"timeout\":\"1nanos\",\"indices\":[\""
                + prefix
                + "_pricing\",\""
                + prefix
                + "_tickets\"]}"
        );
        Map<String, Object> result = parseResponseToMap(response);
        assertEquals(true, result.get("timed_out"));
        Map<String, Object> trace = (Map<String, Object>) result.get("trace");
        assertEquals(2, ((Number) trace.get("resolved_index_count")).intValue());
        assertEquals(List.of("request deadline reached before routing; returned partial results"), trace.get("warnings"));
    }

    @SuppressWarnings("unchecked")
    public void testWildcardSkipsHiddenIndices() throws IOException {
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), true);
        String prefix = createMultiIndices();
        createIndex(prefix + "_hidden", "{\"settings\":{\"index.hidden\":true},\"mappings\":{\"properties\":{\"t\":{\"type\":\"text\"}}}}");
        Map<String, Object> result = parseResponseToMap(
            request(
                "POST",
                "/_plugins/_ml/agentic_index/" + prefix + "_*/_retrieve",
                "{\"question\":\"q\",\"llm_model_id\":\"m\",\"timeout\":\"1nanos\"}"
            )
        );
        assertEquals(3, ((Number) ((Map<String, Object>) result.get("trace")).get("resolved_index_count")).intValue());
    }

    public void testRejectsTargetAboveMaxIndices() throws IOException {
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), true);
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_MAX_INDICES.getKey(), 2);
        String prefix = createMultiIndices();
        ResponseException e = expectThrows(
            ResponseException.class,
            () -> request("POST", "/_plugins/_ml/agentic_index/" + prefix + "_*/_ask", "{\"question\":\"q\",\"llm_model_id\":\"m\"}")
        );
        assertEquals(400, e.getResponse().getStatusLine().getStatusCode());
        assertTrue(e.getMessage(), e.getMessage().contains("resolves to more than 2 indices that you can read"));
    }

    @SuppressWarnings("unchecked")
    public void testAskRoutesAcrossSeveralIndices() throws Exception {
        String prefix = createMultiIndices();
        String modelId = registerMockModel();
        mockLlm.respondWith(prompt -> {
            if (AgenticIndexMockLlm.isRouting(prompt)) {
                return "{\"indices\":[{\"index\":\""
                    + prefix
                    + "_pricing\",\"reason\":\"plans\"},{\"index\":\""
                    + prefix
                    + "_tickets\",\"reason\":\"incidents\"}]}";
            }
            if (AgenticIndexMockLlm.isPlanning(prompt)) {
                return "{\"subqueries\":[{\"index\":\""
                    + prefix
                    + "_pricing\",\"query\":\"sso\",\"filters\":{\"plan\":\"Pro\"}},"
                    + "{\"index\":\""
                    + prefix
                    + "_tickets\",\"query\":\"sso outage\"}]}";
            }
            if (AgenticIndexMockLlm.isSufficiency(prompt)) {
                return "{\"sufficient\":true,\"relevant\":[\"P1\",\"P2\"]}";
            }
            return "SSO is in Pro [P1]. The SSO outage was high severity [P2].";
        });

        Map<String, Object> result = parseResponseToMap(
            request(
                "POST",
                "/_plugins/_ml/agentic_index/" + prefix + "_pricing," + prefix + "_tickets," + prefix + "_hr/_ask",
                "{\"question\":\"Which plan has SSO and was there an SSO outage?\",\"llm_model_id\":\""
                    + modelId
                    + "\",\"max_indices_selected\":2}"
            )
        );

        assertEquals("SSO is in Pro [P1]. The SSO outage was high severity [P2].", result.get("answer"));
        List<Map<String, Object>> citations = (List<Map<String, Object>>) result.get("citations");
        assertEquals(List.of(prefix + "_pricing", prefix + "_tickets"), citations.stream().map(c -> c.get("index")).toList());
        Map<String, Object> trace = (Map<String, Object>) result.get("trace");
        assertEquals(3, ((Number) trace.get("resolved_index_count")).intValue());
        Map<String, Object> routing = (Map<String, Object>) trace.get("routing");
        assertEquals(true, routing.get("routed"));
        assertEquals(3, ((Number) routing.get("catalog_index_count")).intValue());
        List<Map<String, Object>> selected = (List<Map<String, Object>>) routing.get("selected");
        assertEquals(List.of(prefix + "_pricing", prefix + "_tickets"), selected.stream().map(e -> e.get("index")).toList());
        assertEquals("plans", selected.get(0).get("reason"));
        List<Map<String, Object>> subQueries = (List<Map<String, Object>>) ((List<Map<String, Object>>) trace.get("rounds"))
            .get(0)
            .get("subqueries");
        assertEquals(List.of(prefix + "_pricing", prefix + "_tickets"), subQueries.stream().map(q -> q.get("index")).toList());

        String routingPrompt = mockLlm.prompts.get(0);
        for (String name : List.of("_pricing", "_tickets", "_hr")) {
            assertTrue(routingPrompt, routingPrompt.contains("- " + prefix + name + ": fields"));
        }
        // The catalog comes from mappings; field statistics are collected only for the selected indices.
        assertFalse(routingPrompt, routingPrompt.contains("plan=Pro"));
        String planPrompt = mockLlm.prompts.get(1);
        assertTrue(planPrompt, planPrompt.contains("- plan values: Pro"));
        assertTrue(planPrompt.contains("Index: " + prefix + "_tickets"));
        assertFalse(planPrompt.contains("Index: " + prefix + "_hr"));
    }

    @SuppressWarnings("unchecked")
    public void testRetrieveWithWildcard() throws Exception {
        String prefix = createMultiIndices();
        String modelId = registerMockModel();
        mockLlm.respondWith(prompt -> {
            if (AgenticIndexMockLlm.isPlanning(prompt)) {
                return "{\"subqueries\":[{\"index\":\""
                    + prefix
                    + "_tickets\",\"query\":\"outage\"},"
                    + "{\"index\":\""
                    + prefix
                    + "_hr\",\"query\":\"offsite\"}]}";
            }
            return "{\"sufficient\":true}";
        });

        Map<String, Object> result = parseResponseToMap(
            request(
                "POST",
                "/_plugins/_ml/agentic_index/" + prefix + "_*/_retrieve",
                "{\"question\":\"Any outages or offsites?\",\"llm_model_id\":\"" + modelId + "\"}"
            )
        );

        assertFalse(result.containsKey("answer"));
        List<Map<String, Object>> passages = (List<Map<String, Object>>) result.get("passages");
        assertEquals(Set.of(prefix + "_tickets", prefix + "_hr"), passages.stream().map(p -> p.get("index")).collect(Collectors.toSet()));
        Map<String, Object> trace = (Map<String, Object>) result.get("trace");
        assertEquals(3, ((Number) trace.get("resolved_index_count")).intValue());
        Map<String, Object> routing = (Map<String, Object>) trace.get("routing");
        assertEquals(false, routing.get("routed"));
        assertEquals(3, ((List<?>) routing.get("selected")).size());
        assertEquals(2, ((Number) trace.get("llm_call_count")).intValue());
    }

    // Three indices with different fields; returns their common name prefix.
    private String createMultiIndices() throws IOException {
        String prefix = index + "_m";
        createIndex(prefix + "_pricing", "{\"mappings\":{\"properties\":{\"title\":{\"type\":\"text\"},\"plan\":{\"type\":\"keyword\"}}}}");
        createIndex(
            prefix + "_tickets",
            "{\"mappings\":{\"properties\":{\"summary\":{\"type\":\"text\"},\"severity\":{\"type\":\"keyword\"},"
                + "\"opened_at\":{\"type\":\"date\"}}}}"
        );
        createIndex(prefix + "_hr", "{\"mappings\":{\"properties\":{\"note\":{\"type\":\"text\"}}}}");
        request("PUT", "/" + prefix + "_pricing/_doc/1?refresh=true", "{\"title\":\"SSO is included in Pro\",\"plan\":\"Pro\"}");
        request(
            "PUT",
            "/" + prefix + "_tickets/_doc/1?refresh=true",
            "{\"summary\":\"SSO outage for Pro customers\",\"severity\":\"high\",\"opened_at\":\"2026-10-02\"}"
        );
        request("PUT", "/" + prefix + "_hr/_doc/1?refresh=true", "{\"note\":\"Team offsite in March\"}");
        return prefix;
    }

    private void createIndex(String name, String body) throws IOException {
        extraIndices.add(name);
        request("PUT", "/" + name, body);
    }

    private String registerMockModel() throws Exception {
        Assume.assumeFalse("Containerized node cannot reach the test-JVM loopback mock LLM", AgenticIndexMockLlm.unreachableFromCluster());
        mockLlm = new AgenticIndexMockLlm();
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), true);
        updateClusterSettings("plugins.ml_commons.trusted_connector_endpoints_regex", List.of("^.*$"));
        updateClusterSettings("plugins.ml_commons.connector.private_ip_enabled", true);
        String connectorId = registerConnector(mockLlm.connectorBody());
        String name = "agentic_index_mock_" + randomAlphaOfLength(6).toLowerCase(Locale.ROOT);
        Response response = RestMLRemoteInferenceIT.registerRemoteModel(name, name, connectorId);
        String modelId = (String) parseResponseToMap(response).get("model_id");
        response = RestMLRemoteInferenceIT.deployRemoteModel(modelId);
        waitForTask((String) parseResponseToMap(response).get("task_id"), MLTaskState.COMPLETED);
        return modelId;
    }

    public void testRejectsUnknownField() throws IOException {
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), true);
        ResponseException e = expectThrows(
            ResponseException.class,
            () -> request("POST", "/_plugins/_ml/agentic_index/" + index + "/_retrieve", "{\"questoin\":\"q\"}")
        );
        assertEquals(400, e.getResponse().getStatusLine().getStatusCode());
        assertTrue(e.getMessage(), e.getMessage().contains("Unknown field [questoin]"));
    }

    public void testRejectsInvalidRequest() throws IOException {
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), true);
        ResponseException e = expectThrows(
            ResponseException.class,
            () -> request("POST", "/_plugins/_ml/agentic_index/" + index + "/_ask", "{\"question\":\"q\",\"llm_model_id\":\"m\",\"k\":0}")
        );
        assertEquals(400, e.getResponse().getStatusLine().getStatusCode());
        assertTrue(e.getMessage(), e.getMessage().contains("k must be between 1 and 20"));
    }

    public void testUnknownModelFailsAfterProfiling() throws IOException {
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), true);
        ResponseException e = expectThrows(
            ResponseException.class,
            () -> request(
                "POST",
                "/_plugins/_ml/agentic_index/" + index + "/_ask",
                "{\"question\":\"Which plans include SSO?\",\"llm_model_id\":\"no-such-model\"}"
            )
        );
        // The failure comes from the model lookup in the predict call, so index profiling succeeded.
        assertTrue(e.getMessage(), e.getMessage().contains("no-such-model") || e.getMessage().contains(".plugins-ml-model"));
    }

    public void testUnknownIndex() throws IOException {
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), true);
        ResponseException e = expectThrows(
            ResponseException.class,
            () -> request("POST", "/_plugins/_ml/agentic_index/no_such_index_x/_ask", "{\"question\":\"q\",\"llm_model_id\":\"m\"}")
        );
        assertEquals(404, e.getResponse().getStatusLine().getStatusCode());
    }

    private Response request(String method, String endpoint, String body) throws IOException {
        return TestHelper.makeRequest(client(), method, endpoint, ImmutableMap.of(), body, null);
    }
}
