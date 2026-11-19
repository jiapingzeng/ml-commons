/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.ml.common.utils.StringUtils.gson;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetAddress;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.CopyOnWriteArrayList;
import java.util.function.Function;

import com.sun.net.httpserver.HttpExchange;
import com.sun.net.httpserver.HttpServer;

/**
 * Chat-completions mock for the agentic index ITs. It records every user prompt and answers with {@code responder}.
 * It listens on the test JVM's loopback interface, so the cluster must run on the same host network.
 */
final class AgenticIndexMockLlm implements AutoCloseable {

    final List<String> prompts = new CopyOnWriteArrayList<>();
    private final HttpServer server;
    private volatile Function<String, String> responder = prompt -> "{}";

    AgenticIndexMockLlm() throws IOException {
        server = HttpServer.create(new InetSocketAddress(InetAddress.getLoopbackAddress(), 0), 0);
        server.createContext("/", this::handle);
        server.start();
    }

    /** Containerized clusters with a published port cannot reach the test JVM's loopback mock. */
    static boolean unreachableFromCluster() {
        return "docker-cluster".equals(System.getProperty("tests.clustername"));
    }

    void respondWith(Function<String, String> responder) {
        this.responder = responder;
    }

    String connectorBody() {
        return "{\"name\":\"mock chat\",\"version\":1,\"protocol\":\"http\",\"parameters\":{},\"credential\":{\"key\":\"unused\"},"
            + "\"actions\":[{\"action_type\":\"predict\",\"method\":\"POST\",\"url\":\""
            + String.format(Locale.ROOT, "http://127.0.0.1:%d/v1/chat/completions", server.getAddress().getPort())
            + "\",\"headers\":{\"Authorization\":\"Bearer ${credential.key}\"},"
            + "\"request_body\":\"{\\\"messages\\\": ${parameters.messages}, \\\"max_tokens\\\": ${parameters.max_tokens}}\"}]}";
    }

    @SuppressWarnings("unchecked")
    private void handle(HttpExchange exchange) throws IOException {
        String reply;
        try (InputStream in = exchange.getRequestBody()) {
            Map<String, Object> body = gson.fromJson(new String(in.readAllBytes(), StandardCharsets.UTF_8), Map.class);
            List<Map<String, Object>> messages = (List<Map<String, Object>>) body.get("messages");
            String prompt = String.valueOf(messages.get(messages.size() - 1).get("content"));
            prompts.add(prompt);
            reply = responder.apply(prompt);
        } catch (RuntimeException e) {
            reply = "mock error: " + e.getMessage();
        }
        Map<String, Object> completion = Map
            .of(
                "choices",
                List.of(Map.of("message", Map.of("role", "assistant", "content", reply))),
                "usage",
                Map.of("prompt_tokens", 10, "completion_tokens", 5)
            );
        byte[] bytes = gson.toJson(completion).getBytes(StandardCharsets.UTF_8);
        exchange.getResponseHeaders().add("Content-Type", "application/json");
        exchange.sendResponseHeaders(200, bytes.length);
        try (OutputStream out = exchange.getResponseBody()) {
            out.write(bytes);
        }
    }

    static boolean isRouting(String prompt) {
        return prompt.contains("Index catalog (one line per index");
    }

    static boolean isPlanning(String prompt) {
        return prompt.contains("Plan the smallest set of independent search subqueries");
    }

    static boolean isSufficiency(String prompt) {
        return prompt.contains("Judge whether the retrieved passages");
    }

    @Override
    public void close() {
        server.stop(0);
    }
}
