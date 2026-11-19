/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexRequest;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexRequest.Mode;
import org.opensearch.rest.RestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.test.rest.FakeRestRequest;

public class RestMLAgenticIndexActionTests extends OpenSearchTestCase {

    private MLFeatureEnabledSetting featureSetting;
    private RestMLAgenticIndexAction restAction;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        featureSetting = mock(MLFeatureEnabledSetting.class);
        when(featureSetting.isAgenticIndexEnabled()).thenReturn(true);
        restAction = new RestMLAgenticIndexAction(featureSetting);
    }

    public void testNameAndRoutes() {
        assertEquals("ml_agentic_index_action", restAction.getName());
        List<RestHandler.Route> routes = restAction.routes();
        assertEquals(4, routes.size());
        assertEquals(RestRequest.Method.POST, routes.get(0).getMethod());
        assertEquals("/_plugins/_ml/agentic_index/{index}/_ask", routes.get(0).getPath());
        assertEquals("/_plugins/_ml/agentic_index/{index}/_retrieve", routes.get(1).getPath());
        assertEquals("/_plugins/_ml/agentic_index/_ask", routes.get(2).getPath());
        assertEquals("/_plugins/_ml/agentic_index/_retrieve", routes.get(3).getPath());
    }

    public void testIndicesInBody() throws IOException {
        RestRequest rest = new FakeRestRequest.Builder(NamedXContentRegistry.EMPTY)
            .withMethod(RestRequest.Method.POST)
            .withPath("/_plugins/_ml/agentic_index/_retrieve")
            .withContent(new BytesArray("{\"question\":\"q\",\"llm_model_id\":\"m\",\"indices\":[\"a\",\"b-*\"]}"), MediaTypeRegistry.JSON)
            .build();
        MLAgenticIndexRequest request = restAction.getRequest(rest);
        assertEquals(Mode.RETRIEVE, request.getMode());
        assertEquals("a,b-*", request.getIndex());
    }

    public void testCommaSeparatedPathIndex() throws IOException {
        RestRequest rest = new FakeRestRequest.Builder(NamedXContentRegistry.EMPTY)
            .withMethod(RestRequest.Method.POST)
            .withPath("/_plugins/_ml/agentic_index/a,b/_ask")
            .withParams(new java.util.HashMap<>(Map.of("index", "a,b")))
            .withContent(new BytesArray("{\"question\":\"q\",\"llm_model_id\":\"m\"}"), MediaTypeRegistry.JSON)
            .build();
        assertEquals("a,b", restAction.getRequest(rest).getIndex());
    }

    public void testAskRequest() throws IOException {
        MLAgenticIndexRequest request = restAction.getRequest(rest("/_ask", "{\"question\":\"q\",\"llm_model_id\":\"m\",\"k\":3}"));
        assertEquals(Mode.ASK, request.getMode());
        assertEquals("support-docs", request.getIndex());
        assertEquals("q", request.getQuestion());
        assertEquals(3, request.getK());
        assertNull(request.getTenantId());
    }

    public void testRetrieveRequest() throws IOException {
        MLAgenticIndexRequest request = restAction.getRequest(rest("/_retrieve", "{\"question\":\"q\",\"llm_model_id\":\"m\"}"));
        assertEquals(Mode.RETRIEVE, request.getMode());
    }

    public void testDisabled() {
        when(featureSetting.isAgenticIndexEnabled()).thenReturn(false);
        OpenSearchStatusException e = expectThrows(OpenSearchStatusException.class, () -> restAction.getRequest(rest("/_ask", "{}")));
        assertEquals(RestStatus.FORBIDDEN, e.status());
        assertTrue(e.getMessage().contains("plugins.ml_commons.agentic_index_enabled"));
    }

    public void testMissingBody() {
        RestRequest request = new FakeRestRequest.Builder(NamedXContentRegistry.EMPTY)
            .withPath("/_plugins/_ml/agentic_index/support-docs/_ask")
            .withParams(Map.of("index", "support-docs"))
            .build();
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> restAction.getRequest(request));
        assertEquals("Request body is required", e.getMessage());
    }

    private RestRequest rest(String suffix, String body) {
        return new FakeRestRequest.Builder(NamedXContentRegistry.EMPTY)
            .withMethod(RestRequest.Method.POST)
            .withPath("/_plugins/_ml/agentic_index/support-docs" + suffix)
            .withParams(new java.util.HashMap<>(Map.of("index", "support-docs")))
            .withContent(new BytesArray(body), MediaTypeRegistry.JSON)
            .build();
    }
}
