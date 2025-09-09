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
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.atomic.AtomicBoolean;

import org.apache.commons.lang3.StringUtils;
import org.opensearch.core.action.ActionListener;
import org.opensearch.ml.common.spi.tools.Parser;
import org.opensearch.ml.common.spi.tools.Tool;
import org.opensearch.ml.common.spi.tools.ToolAnnotation;
import org.opensearch.ml.common.utils.ToolUtils;
import org.opensearch.ml.engine.algorithms.remote.ConnectorUtils;
import org.opensearch.ml.engine.httpclient.MLHttpClientFactory;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonSyntaxException;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.log4j.Log4j2;
import software.amazon.awssdk.core.internal.http.async.SimpleHttpContentPublisher;
import software.amazon.awssdk.core.sync.RequestBody;
import software.amazon.awssdk.http.SdkHttpFullRequest;
import software.amazon.awssdk.http.SdkHttpMethod;
import software.amazon.awssdk.http.SdkHttpResponse;
import software.amazon.awssdk.http.async.AsyncExecuteRequest;
import software.amazon.awssdk.http.async.SdkAsyncHttpClient;
import software.amazon.awssdk.http.async.SdkAsyncHttpResponseHandler;

@Log4j2
@ToolAnnotation(RemoteSearchIndexTool.TYPE)
public class RemoteSearchIndexTool implements Tool {
    public static final String TYPE = "RemoteSearchIndexTool";
    public static final String INPUT_FIELD = "input";
    public static final String INDEX_FIELD = "index";
    public static final String QUERY_FIELD = "query";
    public static final String STRICT_FIELD = "strict";
    public static final String RETURN_RAW_RESPONSE = "return_raw_response";

    public static final String DEFAULT_DESCRIPTION =
        "Searches an index using a query written in query domain-specific language (DSL) in OpenSearch. Required arguments: 1. `index`, the OpenSearch index name to search. 2. `query`, the OpenSearch DSL formatted query. Returns documents matching the query in the provided index from the remote cluster.";

    public static final String DEFAULT_INPUT_SCHEMA = """
        {
            "type": "object",
            "properties": {
                "index": {
                    "type": "string",
                    "description": "OpenSearch index name to search"
                },
                "query": {
                    "type": "object",
                    "description": "OpenSearch search index query in DSL format. Must be a valid OpenSearch query."
                },
                "input": {
                    "type": "string",
                    "description": "Alternative input format containing index and query as JSON string"
                },
                "return_raw_response": {
                    "type": "boolean",
                    "description": "Whether to return the full search response (default: false)"
                }
            },
            "required": ["query"],
            "additionalProperties": false
        }""";

    public static final Map<String, Object> DEFAULT_ATTRIBUTES = Map.of(TOOL_INPUT_SCHEMA_FIELD, DEFAULT_INPUT_SCHEMA, STRICT_FIELD, false);

    private static final Gson GSON = new GsonBuilder().serializeSpecialFloatingPointValues().create();

    @Setter
    @Getter
    private String name = RemoteSearchIndexTool.TYPE;
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

    public RemoteSearchIndexTool(
        String endpoint,
        String accessKey,
        String secretKey,
        String sessionToken,
        String region,
        String serviceName
    ) {
        if (StringUtils.isBlank(endpoint)) {
            throw new IllegalArgumentException("endpoint is required for RemoteSearchIndexTool");
        }
        if (StringUtils.isBlank(accessKey) || StringUtils.isBlank(secretKey) || StringUtils.isBlank(sessionToken)) {
            throw new IllegalArgumentException(
                "AWS credentials (access_key, secret_key, session_token) are required for RemoteSearchIndexTool"
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

        this.attributes = new HashMap<>();
        attributes.put(TOOL_INPUT_SCHEMA_FIELD, DEFAULT_INPUT_SCHEMA);
        attributes.put(STRICT_FIELD, false);
    }

    @Override
    public <T> void run(Map<String, String> originalParameters, ActionListener<T> listener) {
        try {
            Map<String, String> parameters = ToolUtils.extractInputParameters(originalParameters, attributes);

            if (StringUtils.isBlank(this.endpoint)) {
                listener.onFailure(new IllegalArgumentException("RemoteSearchIndexTool was not properly configured with endpoint"));
                return;
            }

            String input = parameters.get(INPUT_FIELD);
            String index = null;
            String query = null;
            boolean returnRawResponse = Boolean.parseBoolean(parameters.getOrDefault(RETURN_RAW_RESPONSE, "false"));

            if (StringUtils.isNotBlank(input)) {
                try {
                    JsonObject jsonObject = GSON.fromJson(input, JsonObject.class);
                    if (jsonObject != null && jsonObject.has(INDEX_FIELD) && jsonObject.has(QUERY_FIELD)) {
                        index = jsonObject.get(INDEX_FIELD).getAsString();
                        JsonElement queryElement = jsonObject.get(QUERY_FIELD);
                        query = queryElement == null ? null : queryElement.toString();
                    }
                } catch (JsonSyntaxException e) {
                    log.error("Invalid JSON input: {}", input, e);
                    listener.onFailure(new IllegalArgumentException("Invalid JSON input: " + e.getMessage()));
                    return;
                }
            }

            if (StringUtils.isBlank(index)) {
                index = parameters.get(INDEX_FIELD);
            }
            if (StringUtils.isBlank(query)) {
                query = parameters.get(QUERY_FIELD);
            }

            if (StringUtils.isBlank(index) || StringUtils.isBlank(query)) {
                listener
                    .onFailure(
                        new IllegalArgumentException("RemoteSearchIndexTool requires both 'index' and 'query' parameters in valid format")
                    );
                return;
            }

            String searchUrl = buildSearchUrl(this.endpoint, index);

            validateEndpoint(searchUrl);

            executeRemoteSearchRequest(
                searchUrl,
                query,
                returnRawResponse,
                this.accessKey,
                this.secretKey,
                this.sessionToken,
                this.region,
                this.serviceName,
                listener
            );

        } catch (Exception e) {
            log.error("Failed to run RemoteSearchIndexTool", e);
            listener.onFailure(e);
        }
    }

    private String buildSearchUrl(String endpoint, String index) {
        StringBuilder url = new StringBuilder();
        url.append(endpoint);
        if (!endpoint.endsWith("/")) {
            url.append("/");
        }
        url.append(index).append("/_search");
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
    private <T> void executeRemoteSearchRequest(
        String searchUrl,
        String query,
        boolean returnRawResponse,
        String accessKey,
        String secretKey,
        String sessionToken,
        String region,
        String serviceName,
        ActionListener<T> listener
    ) {
        try {
            RequestBody requestBody = RequestBody.fromString(query, StandardCharsets.UTF_8);

            SdkHttpFullRequest request = SdkHttpFullRequest
                .builder()
                .method(SdkHttpMethod.POST)
                .uri(java.net.URI.create(searchUrl))
                .contentStreamProvider(requestBody.contentStreamProvider())
                .putHeader("Content-Type", "application/json")
                .putHeader("Content-Length", requestBody.optionalContentLength().get().toString())
                .build();

            request = ConnectorUtils.signRequest(request, accessKey, secretKey, sessionToken, serviceName, region);

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
                                log.error("Failed to read remote search response", throwable);
                                listener.onFailure(new RuntimeException("Failed to read remote search response", throwable));
                            } else {
                                try {
                                    String formattedResponse = formatSearchResponse(responseStr, returnRawResponse);
                                    @SuppressWarnings("unchecked")
                                    T response = (T) formattedResponse;
                                    listener.onResponse(response);
                                } catch (Exception e) {
                                    log.error("Failed to format remote search response", e);
                                    listener.onFailure(e);
                                }
                            }
                        });
                    }

                    @Override
                    public void onError(Throwable error) {
                        log.error("Remote search HTTP request failed", error);
                        listener.onFailure(new RuntimeException("Remote search HTTP request failed", error));
                    }
                })
                .build();

            AccessController.doPrivileged((PrivilegedExceptionAction<CompletableFuture<Void>>) () -> httpClient.execute(executeRequest));

        } catch (Exception e) {
            log.error("Failed to execute remote search HTTP request", e);
            listener.onFailure(e);
        }
    }

    private String formatSearchResponse(String jsonResponse, boolean returnRawResponse) {
        if (StringUtils.isBlank(jsonResponse)) {
            return "No results found or empty response from remote cluster.";
        }

        try {
            if (returnRawResponse) {
                return jsonResponse;
            }

            JsonObject responseObj = gson.fromJson(jsonResponse, JsonObject.class);
            JsonObject hitsObj = responseObj.getAsJsonObject("hits");

            if (hitsObj == null || !hitsObj.has("hits")) {
                return "No search hits found in remote cluster response.";
            }

            JsonElement hitsArray = hitsObj.get("hits");
            if (hitsArray == null || !hitsArray.isJsonArray() || hitsArray.getAsJsonArray().size() == 0) {
                return "";
            }

            StringBuilder contextBuilder = new StringBuilder();
            for (JsonElement hitElement : hitsArray.getAsJsonArray()) {
                JsonObject hit = hitElement.getAsJsonObject();

                Map<String, Object> docContent = new HashMap<>();
                docContent.put("_index", hit.has("_index") ? hit.get("_index").getAsString() : "");
                docContent.put("_id", hit.has("_id") ? hit.get("_id").getAsString() : "");

                double score = 0.0;
                if (hit.has("_score") && !hit.get("_score").isJsonNull()) {
                    score = hit.get("_score").getAsDouble();
                }
                docContent.put("_score", score);

                if (hit.has("_source")) {
                    JsonElement source = hit.get("_source");
                    Map<String, Object> sourceMap = gson.fromJson(source, Map.class);
                    docContent.put("_source", sourceMap);
                }

                String doc = gson.toJson(docContent);
                contextBuilder.append(doc).append("\n");
            }

            return contextBuilder.toString();

        } catch (Exception e) {
            log.error("Failed to parse search response JSON", e);
            return "Error parsing response from remote cluster: " + e.getMessage() + "\nRaw response: " + jsonResponse;
        }
    }

    @Override
    public String getType() {
        return TYPE;
    }

    @Override
    public boolean validate(Map<String, String> parameters) {
        if (parameters == null || parameters.isEmpty()) {
            return false;
        }

        String input = parameters.get(INPUT_FIELD);
        boolean hasInput = StringUtils.isNotBlank(input);
        boolean hasDirectParams = StringUtils.isNotBlank(parameters.get(INDEX_FIELD))
            && StringUtils.isNotBlank(parameters.get(QUERY_FIELD));

        return hasInput || hasDirectParams;
    }

    /**
     * Factory for the {@link RemoteSearchIndexTool}
     */
    public static class Factory implements Tool.Factory<RemoteSearchIndexTool> {
        private static Factory INSTANCE;

        /** 
         * Create or return the singleton factory instance
         */
        public static Factory getInstance() {
            if (INSTANCE != null) {
                return INSTANCE;
            }
            synchronized (RemoteSearchIndexTool.class) {
                if (INSTANCE != null) {
                    return INSTANCE;
                }
                INSTANCE = new Factory();
                return INSTANCE;
            }
        }

        @Override
        public RemoteSearchIndexTool create(Map<String, Object> map) {
            String endpoint = (String) map.get("endpoint");
            String accessKey = (String) map.get("access_key");
            String secretKey = (String) map.get("secret_key");
            String sessionToken = (String) map.get("session_token");
            String region = (String) map.get("region");
            String serviceName = (String) map.get("service_name");

            return new RemoteSearchIndexTool(endpoint, accessKey, secretKey, sessionToken, region, serviceName);
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
