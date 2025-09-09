/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools;

import static org.opensearch.ml.common.CommonValue.TOOL_INPUT_SCHEMA_FIELD;
import static org.opensearch.ml.common.utils.StringUtils.gson;

import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.security.AccessController;
import java.security.PrivilegedExceptionAction;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.commons.lang3.StringUtils;
import org.opensearch.core.action.ActionListener;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.common.spi.tools.Parser;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.spi.tools.ToolAnnotation;
import org.opensearch.ml.common.utils.ToolUtils;
import org.opensearch.ml.engine.algorithms.remote.ConnectorUtils;
import org.opensearch.ml.engine.httpclient.MLHttpClientFactory;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import software.amazon.awssdk.core.internal.http.async.SimpleHttpContentPublisher;
import software.amazon.awssdk.http.SdkHttpFullRequest;
import software.amazon.awssdk.http.SdkHttpMethod;
import software.amazon.awssdk.http.SdkHttpResponse;
import software.amazon.awssdk.http.async.AsyncExecuteRequest;
import software.amazon.awssdk.http.async.SdkAsyncHttpClient;
import software.amazon.awssdk.http.async.SdkAsyncHttpResponseHandler;

@Log4j2
@ToolAnnotation(RemoteIndexMappingTool.TYPE)
public class RemoteIndexMappingTool implements Tool {
    public static final String TYPE = "RemoteIndexMappingTool";
    public static final String STRICT_FIELD = "strict";
    public static final String DEFAULT_DESCRIPTION =
        "Returns index mappings and settings for specified indices in a remote OpenSearch cluster. Optional arguments: 1. `index`, a comma-delimited list of one or more indices (default is an empty list meaning all indices). Use only valid index names. 2. `local`, whether to return information from the local node only instead of the cluster manager node (default is false). Response format: For each index, 'mappings' contains field definitions under 'properties' (each field has a 'type'), and 'settings' contains configuration as a flattened key-value map.";
    public static final String DEFAULT_INPUT_SCHEMA =
        """
            {
                "type": "object",
                "properties": {
                    "index": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
                        "description": "OpenSearch index name list, separated by comma. For example: [\\\"index1\\\", \\\"index2\\\"], use empty array [] to get mappings for all indices in the cluster"
                    },
                    "local": {
                        "type": "boolean",
                        "description": "Whether to return information from the local node only instead of the cluster manager node (default: false)"
                    }
                },
                "required": [],
                "additionalProperties": false
            }""";
    public static final Map<String, Object> DEFAULT_ATTRIBUTES = Map.of(TOOL_INPUT_SCHEMA_FIELD, DEFAULT_INPUT_SCHEMA, STRICT_FIELD, false);

    @Setter
    @Getter
    private String name = RemoteIndexMappingTool.TYPE;
    @Getter
    @Setter
    private String description = DEFAULT_DESCRIPTION;
    @Getter
    @Setter
    private Map<String, Object> attributes;
    @Getter
    private String version;

    @Setter
    private Parser<?, ?> inputParser;
    @Setter
    private Parser<?, ?> outputParser;

    private SdkAsyncHttpClient httpClient;
    private final AtomicBoolean connectorPrivateIpEnabled = new AtomicBoolean(false);

    private final String endpoint;
    private final String accessKey;
    private final String secretKey;
    private final String sessionToken;
    private final String region;
    private final String serviceName;

    public RemoteIndexMappingTool(
        String endpoint,
        String accessKey,
        String secretKey,
        String sessionToken,
        String region,
        String serviceName
    ) {
        if (StringUtils.isBlank(endpoint)) {
            throw new IllegalArgumentException("endpoint is required for RemoteIndexMappingTool");
        }
        if (StringUtils.isBlank(accessKey) || StringUtils.isBlank(secretKey) || StringUtils.isBlank(sessionToken)) {
            throw new IllegalArgumentException(
                "AWS credentials (access_key, secret_key, session_token) are required for RemoteIndexMappingTool"
            );
        }

        this.endpoint = endpoint;
        this.accessKey = accessKey;
        this.secretKey = secretKey;
        this.sessionToken = sessionToken;
        this.region = StringUtils.isNotBlank(region) ? region : "us-east-1";
        this.serviceName = StringUtils.isNotBlank(serviceName) ? serviceName : "es";

        Duration connectionTimeout = Duration.ofSeconds(30);
        Duration readTimeout = Duration.ofSeconds(30);
        Integer maxConnection = 30;
        this.httpClient = MLHttpClientFactory.getAsyncHttpClient(connectionTimeout, readTimeout, maxConnection);

        outputParser = new Parser<>() {
            @Override
            public Object parse(Object o) {
                @SuppressWarnings("unchecked")
                List<ModelTensors> mlModelOutputs = (List<ModelTensors>) o;
                return mlModelOutputs.get(0).getMlModelTensors().get(0).getDataAsMap().get("response");
            }
        };

        this.attributes = new HashMap<>();
        attributes.put(TOOL_INPUT_SCHEMA_FIELD, DEFAULT_INPUT_SCHEMA);
        attributes.put(STRICT_FIELD, false);
    }

    @Override
    public <T> void run(Map<String, String> originalParameters, ActionListener<T> listener) {
        try {
            Map<String, String> parameters = ToolUtils.extractInputParameters(originalParameters, attributes);

            if (StringUtils.isBlank(this.endpoint)) {
                listener.onFailure(new IllegalArgumentException("RemoteIndexMappingTool was not properly configured with endpoint"));
                return;
            }

            List<String> indexList = new ArrayList<>();
            if (StringUtils.isNotBlank(parameters.get("index"))) {
                indexList = parameters.containsKey("index") ? gson.fromJson(parameters.get("index"), List.class) : Collections.emptyList();
            }

            boolean local = parameters.containsKey("local") && Boolean.parseBoolean(parameters.get("local"));

            String indicesParam = indexList.isEmpty() ? "_all" : String.join(",", indexList);
            String requestUrl = buildGetIndexUrl(this.endpoint, indicesParam, local);

            validateEndpoint(requestUrl);

            executeRemoteRequest(
                requestUrl,
                true, // Always use AWS auth since it's configured at creation time
                this.accessKey,
                this.secretKey,
                this.sessionToken,
                this.region,
                this.serviceName,
                listener
            );

        } catch (Exception e) {
            log.error("Failed to run RemoteIndexMappingTool", e);
            listener.onFailure(e);
        }
    }

    private String buildGetIndexUrl(String endpoint, String indices, boolean local) {
        StringBuilder url = new StringBuilder();
        url.append(endpoint);
        if (!endpoint.endsWith("/")) {
            url.append("/");
        }
        url.append(indices);
        if (local) {
            url.append("?local=true");
        }
        return url.toString();
    }

    private void validateEndpoint(String requestUrl) throws Exception {
        URL url = new URL(requestUrl);
        String protocol = url.getProtocol();
        String host = url.getHost();
        int port = url.getPort();
        MLHttpClientFactory.validate(protocol, host, port, connectorPrivateIpEnabled);
    }

    @SuppressWarnings("removal")
    private <T> void executeRemoteRequest(
        String requestUrl,
        boolean useAwsAuth,
        String accessKey,
        String secretKey,
        String sessionToken,
        String region,
        String serviceName,
        ActionListener<T> listener
    ) {
        try {
            SdkHttpFullRequest request = SdkHttpFullRequest
                .builder()
                .method(SdkHttpMethod.GET)
                .uri(java.net.URI.create(requestUrl))
                .build();

            if (useAwsAuth) {
                request = ConnectorUtils.signRequest(request, accessKey, secretKey, sessionToken, serviceName, region);
            }

            AsyncExecuteRequest executeRequest = AsyncExecuteRequest
                .builder()
                .request(request)
                .requestContentPublisher(new SimpleHttpContentPublisher(request))
                .responseHandler(new SdkAsyncHttpResponseHandler() {
                    @Override
                    public void onHeaders(SdkHttpResponse response) {}

                    @Override
                    public void onStream(org.reactivestreams.Publisher<java.nio.ByteBuffer> stream) {
                        CompletableFuture<String> responseBodyFuture = new CompletableFuture<>();

                        stream.subscribe(new org.reactivestreams.Subscriber<java.nio.ByteBuffer>() {
                            private StringBuilder responseBody = new StringBuilder();
                            private org.reactivestreams.Subscription subscription;

                            @Override
                            public void onSubscribe(org.reactivestreams.Subscription subscription) {
                                this.subscription = subscription;
                                subscription.request(Long.MAX_VALUE);
                            }

                            @Override
                            public void onNext(java.nio.ByteBuffer byteBuffer) {
                                byte[] bytes = new byte[byteBuffer.remaining()];
                                byteBuffer.get(bytes);
                                responseBody.append(new String(bytes, StandardCharsets.UTF_8));
                            }

                            @Override
                            public void onError(Throwable t) {
                                responseBodyFuture.completeExceptionally(t);
                            }

                            @Override
                            public void onComplete() {
                                responseBodyFuture.complete(responseBody.toString());
                            }
                        });

                        responseBodyFuture.whenComplete((responseStr, throwable) -> {
                            if (throwable != null) {
                                log.error("Failed to read remote response", throwable);
                                listener.onFailure(new RuntimeException("Failed to read remote response", throwable));
                            } else {
                                try {
                                    String formattedResponse = formatResponse(responseStr);
                                    @SuppressWarnings("unchecked")
                                    T response = (T) formattedResponse;
                                    listener.onResponse(response);
                                } catch (Exception e) {
                                    log.error("Failed to format remote response", e);
                                    listener.onFailure(e);
                                }
                            }
                        });
                    }

                    @Override
                    public void onError(Throwable error) {
                        log.error("Remote HTTP request failed", error);
                        listener.onFailure(new RuntimeException("Remote HTTP request failed", error));
                    }
                })
                .build();

            AccessController.doPrivileged((PrivilegedExceptionAction<CompletableFuture<Void>>) () -> httpClient.execute(executeRequest));

        } catch (Exception e) {
            log.error("Failed to execute remote HTTP request", e);
            listener.onFailure(e);
        }
    }

    private String formatResponse(String jsonResponse) {
        if (StringUtils.isBlank(jsonResponse)) {
            return "No indices found or empty response from remote cluster.";
        }

        try {
            @SuppressWarnings("unchecked")
            Map<String, Object> indicesData = gson.fromJson(jsonResponse, Map.class);

            if (indicesData == null || indicesData.isEmpty()) {
                return "No indices found in the remote cluster.";
            }

            StringBuilder sb = new StringBuilder();

            for (Map.Entry<String, Object> entry : indicesData.entrySet()) {
                String indexName = entry.getKey();
                @SuppressWarnings("unchecked")
                Map<String, Object> indexData = (Map<String, Object>) entry.getValue();

                sb.append("index: ").append(indexName).append("\n\n");

                Object mappingsObj = indexData.get("mappings");
                if (mappingsObj != null) {
                    sb.append("mappings:\n");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> mappings = (Map<String, Object>) mappingsObj;
                    for (Map.Entry<String, Object> mappingEntry : mappings.entrySet()) {
                        sb.append(mappingEntry.getKey()).append("=").append(mappingEntry.getValue()).append('\n');
                    }
                    sb.append("\n\n");
                }

                Object settingsObj = indexData.get("settings");
                if (settingsObj != null) {
                    sb.append("settings:\n");
                    @SuppressWarnings("unchecked")
                    Map<String, Object> settings = (Map<String, Object>) settingsObj;
                    formatSettingsRecursively(settings, sb, "");
                    sb.append("\n\n");
                }
            }

            return sb.toString();

        } catch (Exception e) {
            log.error("Failed to parse JSON response", e);
            return "Error parsing response from remote cluster: " + e.getMessage() + "\nRaw response: " + jsonResponse;
        }
    }

    @SuppressWarnings("unchecked")
    private void formatSettingsRecursively(Map<String, Object> settings, StringBuilder sb, String prefix) {
        for (Map.Entry<String, Object> entry : settings.entrySet()) {
            String key = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
            Object value = entry.getValue();

            if (value instanceof Map) {
                formatSettingsRecursively((Map<String, Object>) value, sb, key);
            } else {
                sb.append(key).append("=").append(value).append('\n');
            }
        }
    }

    @Override
    public String getType() {
        return TYPE;
    }

    @Override
    public boolean validate(Map<String, String> parameters) {
        return true;
    }

    /**
     * Factory for the {@link RemoteIndexMappingTool}
     */
    public static class Factory implements Tool.Factory<RemoteIndexMappingTool> {
        private static Factory INSTANCE;

        /** 
         * Create or return the singleton factory instance
         */
        public static Factory getInstance() {
            if (INSTANCE != null) {
                return INSTANCE;
            }
            synchronized (RemoteIndexMappingTool.class) {
                if (INSTANCE != null) {
                    return INSTANCE;
                }
                INSTANCE = new Factory();
                return INSTANCE;
            }
        }

        @Override
        public RemoteIndexMappingTool create(Map<String, Object> map) {
            String endpoint = (String) map.get("endpoint");
            String accessKey = (String) map.get("access_key");
            String secretKey = (String) map.get("secret_key");
            String sessionToken = (String) map.get("session_token");
            String region = (String) map.get("region");
            String serviceName = (String) map.get("service_name");

            return new RemoteIndexMappingTool(endpoint, accessKey, secretKey, sessionToken, region, serviceName);
        }

        @Override
        public String getDefaultDescription() {
            return DEFAULT_DESCRIPTION;
        }

        @Override
        public String getDefaultType() {
            return TYPE;
        }

        @Override
        public String getDefaultVersion() {
            return null;
        }

        @Override
        public Map<String, Object> getDefaultAttributes() {
            return DEFAULT_ATTRIBUTES;
        }
    }
}
