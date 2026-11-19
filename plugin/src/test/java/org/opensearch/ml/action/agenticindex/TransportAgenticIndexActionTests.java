/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.verifyNoInteractions;
import static org.mockito.Mockito.when;
import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_AGENTIC_INDEX_MAX_INDICES;
import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_AGENTIC_INDEX_TIMEOUT;

import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.Version;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.admin.indices.mapping.get.GetMappingsRequest;
import org.opensearch.action.admin.indices.mapping.get.GetMappingsResponse;
import org.opensearch.action.search.SearchRequest;
import org.opensearch.action.search.SearchResponse;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.cluster.node.DiscoveryNode;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.settings.ClusterSettings;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexRequest;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexResponse;
import org.opensearch.tasks.CancellableTask;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.threadpool.ThreadPool;
import org.opensearch.transport.TransportService;
import org.opensearch.transport.client.AdminClient;
import org.opensearch.transport.client.Client;
import org.opensearch.transport.client.IndicesAdminClient;

public class TransportAgenticIndexActionTests extends OpenSearchTestCase {

    private final MLAgenticIndexRequest request = MLAgenticIndexRequest
        .builder()
        .index("docs")
        .mode(MLAgenticIndexRequest.Mode.ASK)
        .question("q")
        .llmModelId("m")
        .build();

    private Client client;
    private MLFeatureEnabledSetting setting;

    @Override
    public void setUp() throws Exception {
        super.setUp();
        client = mock(Client.class);
        ThreadPool threadPool = mock(ThreadPool.class);
        when(client.threadPool()).thenReturn(threadPool);
        when(threadPool.getThreadContext()).thenReturn(new ThreadContext(Settings.EMPTY));
        setting = mock(MLFeatureEnabledSetting.class);
        when(setting.isAgenticIndexEnabled()).thenReturn(true);
    }

    private TransportAgenticIndexAction action() {
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.getClusterSettings())
            .thenReturn(
                new ClusterSettings(Settings.EMPTY, Set.of(ML_COMMONS_AGENTIC_INDEX_TIMEOUT, ML_COMMONS_AGENTIC_INDEX_MAX_INDICES))
            );
        when(clusterService.localNode()).thenReturn(new DiscoveryNode("node-1", buildNewFakeTransportAddress(), Version.CURRENT));
        when(clusterService.state()).thenReturn(ClusterState.EMPTY_STATE);
        return new TransportAgenticIndexAction(
            mock(TransportService.class),
            mock(ActionFilters.class),
            client,
            clusterService,
            setting,
            new IndexProfileCache(index -> null, IndexProfileCache.TTL),
            new IndexResolver(client, clusterService::state, new IndexNameExpressionResolver(new ThreadContext(Settings.EMPTY)))
        );
    }

    private Exception executeExpectingFailure(ActionRequest actionRequest) {
        AtomicReference<Exception> failure = new AtomicReference<>();
        action()
            .doExecute(mock(CancellableTask.class), actionRequest, ActionListener.<MLAgenticIndexResponse>wrap(r -> fail(), failure::set));
        assertNotNull(failure.get());
        return failure.get();
    }

    public void testInjectedConstructor() {
        assertNotNull(
            new TransportAgenticIndexAction(
                mock(TransportService.class),
                mock(ActionFilters.class),
                client,
                mock(ClusterService.class),
                setting,
                mock(IndexNameExpressionResolver.class)
            )
        );
    }

    public void testDisabledReturnsForbidden() {
        when(setting.isAgenticIndexEnabled()).thenReturn(false);
        OpenSearchStatusException e = (OpenSearchStatusException) executeExpectingFailure(request);
        assertEquals(RestStatus.FORBIDDEN, e.status());
        verifyNoInteractions(client);
    }

    public void testMultiTenancyRequiresTenant() {
        when(setting.isMultiTenancyEnabled()).thenReturn(true);
        executeExpectingFailure(request);
        verifyNoInteractions(client);
    }

    public void testEnabledStartsRunner() {
        when(client.admin()).thenThrow(new IllegalStateException("admin client used"));
        assertEquals("admin client used", executeExpectingFailure(request).getMessage());
    }

    public void testConvertsGenericActionRequest() {
        when(client.admin()).thenThrow(new IllegalStateException("admin client used"));
        ActionRequest generic = new ActionRequest() {
            @Override
            public org.opensearch.action.ActionRequestValidationException validate() {
                return null;
            }

            @Override
            public void writeTo(StreamOutput out) throws java.io.IOException {
                request.writeTo(out);
            }
        };
        assertEquals("admin client used", executeExpectingFailure(generic).getMessage());
    }

    @SuppressWarnings("unchecked")
    public void testRequestTimeoutAppliesDeadline() {
        AdminClient admin = mock(AdminClient.class);
        IndicesAdminClient indices = mock(IndicesAdminClient.class);
        when(client.admin()).thenReturn(admin);
        when(admin.indices()).thenReturn(indices);
        doAnswer(invocation -> {
            ActionListener<GetMappingsResponse> listener = invocation.getArgument(1);
            GetMappingsResponse response = mock(GetMappingsResponse.class);
            when(response.getMappings())
                .thenReturn(Map.of("docs", new MappingMetadata("docs", Map.of("properties", IndexProfileTests.PROPERTIES))));
            listener.onResponse(response);
            return null;
        }).when(indices).getMappings(any(GetMappingsRequest.class), any());
        doAnswer(invocation -> {
            ActionListener<SearchResponse> listener = invocation.getArgument(1);
            listener.onResponse(mock(SearchResponse.class));
            return null;
        }).when(client).search(any(SearchRequest.class), any());
        MLAgenticIndexRequest timed = MLAgenticIndexRequest
            .builder()
            .index("docs")
            .mode(MLAgenticIndexRequest.Mode.ASK)
            .question("q")
            .llmModelId("m")
            .timeout(TimeValue.timeValueNanos(1))
            .build();
        AtomicReference<MLAgenticIndexResponse> response = new AtomicReference<>();

        action().doExecute(mock(CancellableTask.class), timed, ActionListener.wrap(response::set, e -> fail(e.getMessage())));

        assertEquals(true, response.get().getResult().get("timed_out"));
    }
}
