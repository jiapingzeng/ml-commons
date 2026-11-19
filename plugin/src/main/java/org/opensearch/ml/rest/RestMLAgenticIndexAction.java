/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_AGENTIC_INDEX_DISABLED_MESSAGE;
import static org.opensearch.ml.plugin.MachineLearningPlugin.ML_BASE_URI;
import static org.opensearch.ml.utils.TenantAwareHelper.getTenantID;

import java.io.IOException;
import java.util.List;
import java.util.Locale;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.settings.MLFeatureEnabledSetting;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexAction;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexRequest;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexRequest.Mode;
import org.opensearch.rest.BaseRestHandler;
import org.opensearch.rest.RestRequest;
import org.opensearch.rest.action.RestCancellableNodeClient;
import org.opensearch.rest.action.RestToXContentListener;
import org.opensearch.transport.client.node.NodeClient;

import com.google.common.annotations.VisibleForTesting;
import com.google.common.collect.ImmutableList;

/**
 * Experimental agentic index APIs: _ask returns a cited answer, _retrieve returns cited passages only. The target is the
 * {index} path parameter (an index, alias, pattern or comma-separated list) or the "indices" body field.
 */
public class RestMLAgenticIndexAction extends BaseRestHandler {
    private static final String ACTION_NAME = "ml_agentic_index_action";
    static final String PARAMETER_INDEX = "index";
    static final String ASK_SUFFIX = "/_ask";
    static final String RETRIEVE_SUFFIX = "/_retrieve";

    private final MLFeatureEnabledSetting mlFeatureEnabledSetting;

    public RestMLAgenticIndexAction(MLFeatureEnabledSetting mlFeatureEnabledSetting) {
        this.mlFeatureEnabledSetting = mlFeatureEnabledSetting;
    }

    @Override
    public String getName() {
        return ACTION_NAME;
    }

    @Override
    public List<Route> routes() {
        String base = String.format(Locale.ROOT, "%s/agentic_index/{%s}", ML_BASE_URI, PARAMETER_INDEX);
        // Without an index in the path, the body lists the target indices.
        String bodyTarget = ML_BASE_URI + "/agentic_index";
        return ImmutableList
            .of(
                new Route(RestRequest.Method.POST, base + ASK_SUFFIX),
                new Route(RestRequest.Method.POST, base + RETRIEVE_SUFFIX),
                new Route(RestRequest.Method.POST, bodyTarget + ASK_SUFFIX),
                new Route(RestRequest.Method.POST, bodyTarget + RETRIEVE_SUFFIX)
            );
    }

    @Override
    protected RestChannelConsumer prepareRequest(RestRequest request, NodeClient client) throws IOException {
        MLAgenticIndexRequest agenticIndexRequest = getRequest(request);
        // The cancellable client cancels the task when the HTTP client disconnects.
        return channel -> new RestCancellableNodeClient(client, request.getHttpChannel())
            .execute(MLAgenticIndexAction.INSTANCE, agenticIndexRequest, new RestToXContentListener<>(channel));
    }

    @VisibleForTesting
    MLAgenticIndexRequest getRequest(RestRequest request) throws IOException {
        if (!mlFeatureEnabledSetting.isAgenticIndexEnabled()) {
            throw new OpenSearchStatusException(ML_COMMONS_AGENTIC_INDEX_DISABLED_MESSAGE, RestStatus.FORBIDDEN);
        }
        if (!request.hasContent()) {
            throw new IllegalArgumentException("Request body is required");
        }
        Mode mode = request.path().endsWith(ASK_SUFFIX) ? Mode.ASK : Mode.RETRIEVE;
        String tenantId = getTenantID(mlFeatureEnabledSetting.isMultiTenancyEnabled(), request);
        try (XContentParser parser = request.contentParser()) {
            parser.nextToken();
            return MLAgenticIndexRequest.parse(parser, request.param(PARAMETER_INDEX), mode, tenantId);
        }
    }
}
