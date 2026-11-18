/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import static org.opensearch.commons.ConfigConstants.OPENSEARCH_SECURITY_USER_INFO_THREAD_CONTEXT;
import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_AGENTIC_INDEX_DISABLED_MESSAGE;
import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_AGENTIC_INDEX_TIMEOUT;

import java.time.Clock;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.support.ActionFilters;
import org.opensearch.action.support.HandledTransportAction;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.inject.Inject;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexAction;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexRequest;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexResponse;
import org.opensearch.ml.utils.TenantAwareHelper;
import org.opensearch.tasks.CancellableTask;
import org.opensearch.tasks.Task;
import org.opensearch.transport.TransportService;
import org.opensearch.transport.client.Client;

/**
 * Executes agentic index ask and retrieve requests. The thread context is not stashed: index searches and model
 * predictions are authorized as the calling user.
 */
public class TransportAgenticIndexAction extends HandledTransportAction<ActionRequest, MLAgenticIndexResponse> {

    private final Client client;
    private final ClusterService clusterService;
    private final MLFeatureEnabledSetting mlFeatureEnabledSetting;
    private final IndexProfileCache profileCache;

    @Inject
    public TransportAgenticIndexAction(
        TransportService transportService,
        ActionFilters actionFilters,
        Client client,
        ClusterService clusterService,
        MLFeatureEnabledSetting mlFeatureEnabledSetting
    ) {
        this(transportService, actionFilters, client, clusterService, mlFeatureEnabledSetting, new IndexProfileCache(clusterService));
    }

    TransportAgenticIndexAction(
        TransportService transportService,
        ActionFilters actionFilters,
        Client client,
        ClusterService clusterService,
        MLFeatureEnabledSetting mlFeatureEnabledSetting,
        IndexProfileCache profileCache
    ) {
        super(MLAgenticIndexAction.NAME, transportService, actionFilters, MLAgenticIndexRequest::new);
        this.client = client;
        this.clusterService = clusterService;
        this.mlFeatureEnabledSetting = mlFeatureEnabledSetting;
        this.profileCache = profileCache;
    }

    @Override
    protected void doExecute(Task task, ActionRequest actionRequest, ActionListener<MLAgenticIndexResponse> listener) {
        MLAgenticIndexRequest request = MLAgenticIndexRequest.fromActionRequest(actionRequest);
        if (!mlFeatureEnabledSetting.isAgenticIndexEnabled()) {
            listener.onFailure(new OpenSearchStatusException(ML_COMMONS_AGENTIC_INDEX_DISABLED_MESSAGE, RestStatus.FORBIDDEN));
            return;
        }
        if (!TenantAwareHelper.validateTenantId(mlFeatureEnabledSetting, request.getTenantId(), listener)) {
            return;
        }
        try {
            TimeValue maxTimeout = clusterService.getClusterSettings().get(ML_COMMONS_AGENTIC_INDEX_TIMEOUT);
            TimeValue timeout = request.getTimeout() == null || request.getTimeout().compareTo(maxTimeout) > 0
                ? maxTimeout
                : request.getTimeout();
            String caller = client.threadPool().getThreadContext().getTransient(OPENSEARCH_SECURITY_USER_INFO_THREAD_CONTEXT);
            String localNodeId = clusterService.localNode().getId();
            new AgenticIndexRunner(request, client, Clock.systemUTC(), profileCache, caller, timeout, (CancellableTask) task, localNodeId)
                .run(ActionListener.wrap(result -> listener.onResponse(new MLAgenticIndexResponse(result)), listener::onFailure));
        } catch (Exception e) {
            listener.onFailure(e);
        }
    }
}
