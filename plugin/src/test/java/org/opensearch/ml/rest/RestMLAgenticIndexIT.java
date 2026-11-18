/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_AGENTIC_INDEX_ENABLED;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.junit.After;
import org.junit.Before;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.ml.utils.TestHelper;

import com.google.common.collect.ImmutableMap;

/**
 * Covers the agentic index REST routes against a real cluster without an LLM: the feature gate, request
 * validation, and that a request reaches the model call after profiling the index.
 */
public class RestMLAgenticIndexIT extends MLCommonsRestTestCase {

    private String index;

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
        deleteIndexWithAdminClient(index);
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

    public void testRejectsPatternMatchingSeveralIndices() throws IOException {
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), true);
        request("PUT", "/" + index + "_2", "{\"mappings\":{\"properties\":{\"title\":{\"type\":\"text\"}}}}");
        try {
            ResponseException e = expectThrows(
                ResponseException.class,
                () -> request("POST", "/_plugins/_ml/agentic_index/" + index + "*/_ask", "{\"question\":\"q\",\"llm_model_id\":\"m\"}")
            );
            assertEquals(400, e.getResponse().getStatusLine().getStatusCode());
            assertTrue(e.getMessage(), e.getMessage().contains("resolves to 2 indices"));
        } finally {
            deleteIndexWithAdminClient(index + "_2");
        }
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
