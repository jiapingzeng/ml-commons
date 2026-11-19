/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.rest;

import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_AGENTIC_INDEX_ENABLED;
import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_AGENTIC_INDEX_MAX_INDICES;
import static org.opensearch.ml.common.utils.StringUtils.gson;

import java.io.IOException;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.apache.hc.core5.http.HttpHeaders;
import org.apache.hc.core5.http.HttpHost;
import org.apache.hc.core5.http.message.BasicHeader;
import org.junit.After;
import org.junit.Assume;
import org.junit.Before;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.client.RestClient;
import org.opensearch.commons.rest.SecureRestClientBuilder;
import org.opensearch.ml.common.MLTaskState;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.utils.TestHelper;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

/**
 * Agentic index authorization with the security plugin: every step runs as the caller, so the caller needs read access
 * to the index and access to the LLM's model group in addition to the agentic index cluster permission. A pattern
 * resolves only to the indices the caller can read, and nothing about the other indices reaches the catalog, the
 * results or errors.
 */
public class SecureMLAgenticIndexIT extends MLCommonsRestTestCase {

    private static final String NO_READ_ROLE = "agentic_index_no_index_read";
    private static final String READER_ROLE = "agentic_index_reader";
    private static final String PARTIAL_ROLE = "agentic_index_partial_reader";
    private static final String ALIAS_ROLE = "agentic_index_alias_reader";
    private static final String SECRET_TEXT = "TOPSECRET merger codename Falcon";

    private final String noReadUser = "agentic_index_no_read";
    private final String readerUser = "agentic_index_reader";
    private final String partialUser = "agentic_index_partial";
    private final String aliasUser = "agentic_index_alias";
    private RestClient noReadClient;
    private RestClient readerClient;
    private RestClient partialClient;
    private RestClient aliasClient;
    private String index;
    // Prefix of three indices; the partial reader may read <prefix>_a and <prefix>_b but not <prefix>_secret.
    private String prefix;
    // Prefix of two aliases, <aliasPrefix>_pro and <aliasPrefix>_b; the alias reader may read only those aliases.
    private String aliasPrefix;
    private String modelId;
    private AgenticIndexMockLlm mockLlm;

    @Before
    public void setUpSecurity() throws Exception {
        if (!isHttps()) {
            throw new IllegalArgumentException("Secure Tests are running but HTTPS is not set");
        }
        updateClusterSettings("plugins.ml_commons.model_access_control_enabled", true);
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), true);

        index = "agentic_index_secure_" + randomAlphaOfLength(6).toLowerCase(Locale.ROOT);
        adminRequest("PUT", "/" + index, "{\"mappings\":{\"properties\":{\"title\":{\"type\":\"text\"},\"plan\":{\"type\":\"keyword\"}}}}");
        adminRequest("PUT", "/" + index + "/_doc/1?refresh=true", "{\"title\":\"SSO is in Pro\",\"plan\":\"Pro\"}");

        String clusterPermissions = "\"" + MLAgenticIndexAction.NAME + "\", \"" + MLPredictionTaskAction.NAME + "\"";
        putRole(NO_READ_ROLE, "{\"cluster_permissions\": [" + clusterPermissions + "]}");
        putRole(
            READER_ROLE,
            "{\"cluster_permissions\": ["
                + clusterPermissions
                + "], \"index_permissions\": [{\"index_patterns\": [\""
                + index
                + "\"], \"allowed_actions\": [\"indices:data/read/search\", \"indices:admin/mappings/get\"]}]}"
        );
        noReadClient = createUserClient(noReadUser, NO_READ_ROLE);
        readerClient = createUserClient(readerUser, READER_ROLE);

        prefix = index + "_p";
        adminRequest(
            "PUT",
            "/" + prefix + "_a",
            "{\"mappings\":{\"properties\":{\"title\":{\"type\":\"text\"},\"plan\":{\"type\":\"keyword\"}}}}"
        );
        adminRequest("PUT", "/" + prefix + "_a/_doc/1?refresh=true", "{\"title\":\"SSO is in Pro\",\"plan\":\"Pro\"}");
        adminRequest("PUT", "/" + prefix + "_b", "{\"mappings\":{\"properties\":{\"summary\":{\"type\":\"text\"}}}}");
        adminRequest("PUT", "/" + prefix + "_b/_doc/1?refresh=true", "{\"summary\":\"SSO outage last week\"}");
        adminRequest(
            "PUT",
            "/" + prefix + "_secret",
            "{\"mappings\":{\"properties\":{\"title\":{\"type\":\"text\"},\"codename\":{\"type\":\"keyword\"}}}}"
        );
        adminRequest("PUT", "/" + prefix + "_secret/_doc/1?refresh=true", "{\"title\":\"" + SECRET_TEXT + "\",\"codename\":\"Falcon\"}");
        putRole(
            PARTIAL_ROLE,
            "{\"cluster_permissions\": ["
                + clusterPermissions
                + "], \"index_permissions\": [{\"index_patterns\": [\""
                + prefix
                + "_a\", \""
                + prefix
                + "_b\"], \"allowed_actions\": [\"indices:data/read/search\", \"indices:admin/mappings/get\"]}]}"
        );
        partialClient = createUserClient(partialUser, PARTIAL_ROLE);

        // Aliases over <prefix>_a (filtered to the Pro plan) and <prefix>_b; the alias reader may read only through them.
        aliasPrefix = index + "_al";
        adminRequest("PUT", "/" + prefix + "_a/_doc/2?refresh=true", "{\"title\":\"SSO is in Free\",\"plan\":\"Free\"}");
        adminRequest(
            "POST",
            "/_aliases",
            "{\"actions\":[{\"add\":{\"index\":\""
                + prefix
                + "_a\",\"alias\":\""
                + aliasPrefix
                + "_pro\",\"filter\":{\"term\":{\"plan\":\"Pro\"}}}},{\"add\":{\"index\":\""
                + prefix
                + "_b\",\"alias\":\""
                + aliasPrefix
                + "_b\"}}]}"
        );
        putRole(
            ALIAS_ROLE,
            "{\"cluster_permissions\": ["
                + clusterPermissions
                + "], \"index_permissions\": [{\"index_patterns\": [\""
                + aliasPrefix
                + "_*\"], \"allowed_actions\": [\"indices:data/read/search\", \"indices:admin/mappings/get\"]}]}"
        );
        aliasClient = createUserClient(aliasUser, ALIAS_ROLE);

        // The LLM lives in a private model group owned by the admin, so neither user may predict with it.
        String groupName = "agentic_index_private_" + randomAlphaOfLength(6).toLowerCase(Locale.ROOT);
        Map<String, Object> group = parseResponseToMap(
            adminRequest("POST", "/_plugins/_ml/model_groups/_register", "{\"name\":\"" + groupName + "\",\"access_mode\":\"private\"}")
        );
        String connectorId = registerConnector(
            "{\"name\":\"chat\",\"version\":1,\"protocol\":\"http\",\"parameters\":{\"endpoint\":\"api.openai.com\"},"
                + "\"credential\":{\"key\":\"unused\"},\"actions\":[{\"action_type\":\"predict\",\"method\":\"POST\","
                + "\"url\":\"https://${parameters.endpoint}/v1/chat/completions\",\"headers\":{\"Authorization\":\"Bearer ${credential.key}\"},"
                + "\"request_body\":\"{\\\"messages\\\": ${parameters.messages}}\"}]}"
        );
        Map<String, Object> model = parseResponseToMap(
            adminRequest(
                "POST",
                "/_plugins/_ml/models/_register",
                "{\"name\":\"chat\",\"function_name\":\"remote\",\"model_group_id\":\""
                    + group.get("model_group_id")
                    + "\",\"connector_id\":\""
                    + connectorId
                    + "\"}"
            )
        );
        modelId = (String) model.get("model_id");
    }

    @After
    public void tearDownSecurity() throws IOException {
        noReadClient.close();
        readerClient.close();
        partialClient.close();
        aliasClient.close();
        deleteUser(noReadUser);
        deleteUser(readerUser);
        deleteUser(partialUser);
        deleteUser(aliasUser);
        securityRequest("DELETE", "/_plugins/_security/api/roles/" + NO_READ_ROLE, "");
        securityRequest("DELETE", "/_plugins/_security/api/roles/" + READER_ROLE, "");
        securityRequest("DELETE", "/_plugins/_security/api/roles/" + PARTIAL_ROLE, "");
        securityRequest("DELETE", "/_plugins/_security/api/roles/" + ALIAS_ROLE, "");
        deleteIndexWithAdminClient(index);
        for (String suffix : List.of("_a", "_b", "_secret")) {
            deleteIndexWithAdminClient(prefix + suffix);
        }
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), null);
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_MAX_INDICES.getKey(), null);
        if (mockLlm != null) {
            mockLlm.close();
            updateClusterSettings("plugins.ml_commons.trusted_connector_endpoints_regex", null);
            updateClusterSettings("plugins.ml_commons.connector.private_ip_enabled", null);
        }
    }

    public void testUserWithoutIndexReadIsForbidden() {
        ResponseException e = expectThrows(ResponseException.class, () -> ask(noReadClient));
        assertEquals(403, e.getResponse().getStatusLine().getStatusCode());
        assertTrue(e.getMessage(), e.getMessage().contains("indices:admin/mappings/get"));
    }

    public void testUserWithoutModelGroupAccessIsRefused() {
        ResponseException e = expectThrows(ResponseException.class, () -> ask(readerClient));
        assertEquals(403, e.getResponse().getStatusLine().getStatusCode());
        assertTrue(e.getMessage(), e.getMessage().contains("User doesn't have privilege to perform this operation on this model"));
    }

    @SuppressWarnings("unchecked")
    public void testPatternResolvesOnlyReadableIndices() throws IOException {
        // The cap counts only the indices the caller can read: the partial reader (two indices) gets past resolution and
        // stops at the model it may not use, while the admin (three indices) is rejected.
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_MAX_INDICES.getKey(), 2);
        ResponseException partial = expectThrows(ResponseException.class, () -> ask(partialClient, prefix + "_*"));
        assertEquals(403, partial.getResponse().getStatusLine().getStatusCode());
        assertTrue(
            partial.getMessage(),
            partial.getMessage().contains("User doesn't have privilege to perform this operation on this model")
        );
        ResponseException admin = expectThrows(ResponseException.class, () -> ask(client(), prefix + "_*"));
        assertEquals(400, admin.getResponse().getStatusLine().getStatusCode());
        assertTrue(admin.getMessage(), admin.getMessage().contains("resolves to more than 2 indices that you can read"));
    }

    public void testAliasOnlyReaderPatternResolvesThroughAliases() {
        // Past resolution and the statistics search, the request stops at the model the alias reader may not use.
        ResponseException e = expectThrows(ResponseException.class, () -> ask(aliasClient, aliasPrefix + "_*"));
        assertEquals(403, e.getResponse().getStatusLine().getStatusCode());
        assertTrue(e.getMessage(), e.getMessage().contains("User doesn't have privilege to perform this operation on this model"));
    }

    public void testNamedUnreadableIndexLooksLikeMissingIndex() {
        ResponseException unreadable = expectThrows(
            ResponseException.class,
            () -> retrieve(partialClient, prefix + "_a," + prefix + "_secret")
        );
        ResponseException missing = expectThrows(ResponseException.class, () -> retrieve(partialClient, prefix + "_a," + prefix + "_nope"));
        assertEquals(403, unreadable.getResponse().getStatusLine().getStatusCode());
        assertEquals(403, missing.getResponse().getStatusLine().getStatusCode());
        assertFalse(unreadable.getMessage().contains(SECRET_TEXT));
    }

    public void testPatternWithoutReadableMatchLooksLikeNoMatch() {
        ResponseException unreadable = expectThrows(ResponseException.class, () -> retrieve(partialClient, prefix + "_sec*"));
        ResponseException missing = expectThrows(ResponseException.class, () -> retrieve(partialClient, prefix + "_zzz*"));
        assertEquals(404, unreadable.getResponse().getStatusLine().getStatusCode());
        assertEquals(404, missing.getResponse().getStatusLine().getStatusCode());
        assertTrue(unreadable.getMessage(), unreadable.getMessage().contains("No indices that you can read match [" + prefix + "_sec*]"));
        assertTrue(missing.getMessage(), missing.getMessage().contains("No indices that you can read match [" + prefix + "_zzz*]"));
    }

    @SuppressWarnings("unchecked")
    public void testPartialReaderCatalogAndResultsExcludeUnreadableIndex() throws Exception {
        Assume.assumeFalse("Containerized node cannot reach the test-JVM loopback mock LLM", AgenticIndexMockLlm.unreachableFromCluster());
        String publicModelId = registerPublicMockModel();
        String secret = prefix + "_secret";
        mockLlm.respondWith(prompt -> {
            if (AgenticIndexMockLlm.isRouting(prompt)) {
                // The model names the unreadable index (for example from the question); it must be ignored.
                return "{\"indices\":[{\"index\":\""
                    + secret
                    + "\",\"reason\":\"x\"},{\"index\":\""
                    + prefix
                    + "_a\",\"reason\":\"plans\"}]}";
            }
            if (AgenticIndexMockLlm.isPlanning(prompt)) {
                return "{\"subqueries\":[{\"index\":\""
                    + secret
                    + "\",\"query\":\"falcon\"},{\"index\":\""
                    + prefix
                    + "_a\",\"query\":\"sso\"}]}";
            }
            if (AgenticIndexMockLlm.isSufficiency(prompt)) {
                return "{\"sufficient\":false,\"missing\":\"x\",\"followups\":[{\"index\":\"" + secret + "\",\"query\":\"merger\"}]}";
            }
            return "unexpected";
        });

        Response response = TestHelper
            .makeRequest(
                partialClient,
                "POST",
                "/_plugins/_ml/agentic_index/" + prefix + "_*/_retrieve",
                ImmutableMap.of(),
                "{\"question\":\"Which plan has SSO?\",\"llm_model_id\":\""
                    + publicModelId
                    + "\",\"max_indices_selected\":1,\"max_rounds\":2}",
                null
            );
        Map<String, Object> result = parseResponseToMap(response);

        String routingPrompt = mockLlm.prompts.get(0);
        assertTrue(routingPrompt, routingPrompt.contains("- " + prefix + "_a: fields"));
        assertTrue(routingPrompt, routingPrompt.contains("- " + prefix + "_b: fields"));
        for (String prompt : mockLlm.prompts) {
            assertFalse(prompt, prompt.contains("- " + secret + ":") || prompt.contains("Index: " + secret));
            assertFalse(prompt, prompt.contains(SECRET_TEXT) || prompt.contains("codename"));
        }
        String json = gson.toJson(result);
        assertFalse(json, json.contains(secret) || json.contains(SECRET_TEXT));
        List<Map<String, Object>> passages = (List<Map<String, Object>>) result.get("passages");
        assertFalse(passages.isEmpty());
        passages.forEach(p -> assertEquals(prefix + "_a", p.get("index")));
        Map<String, Object> trace = (Map<String, Object>) result.get("trace");
        assertEquals(2, ((Number) trace.get("resolved_index_count")).intValue());
        Map<String, Object> routing = (Map<String, Object>) trace.get("routing");
        assertEquals(2, ((Number) routing.get("catalog_index_count")).intValue());
        List<Map<String, Object>> selected = (List<Map<String, Object>>) routing.get("selected");
        assertEquals(List.of(prefix + "_a"), selected.stream().map(e -> e.get("index")).toList());
        List<String> warnings = (List<String>) trace.get("warnings");
        assertTrue(warnings.toString(), warnings.contains("routing named 1 indices that are not among the resolved indices; ignored"));
        assertTrue(warnings.toString(), warnings.contains("plan dropped 1 subqueries on indices that are not selected"));
        assertTrue(warnings.toString(), warnings.contains("sufficiency dropped 1 subqueries on indices that are not selected"));
    }

    @SuppressWarnings("unchecked")
    public void testAliasOnlyReaderSearchesThroughAliasFilters() throws Exception {
        Assume.assumeFalse("Containerized node cannot reach the test-JVM loopback mock LLM", AgenticIndexMockLlm.unreachableFromCluster());
        String publicModelId = registerPublicMockModel();
        mockLlm.respondWith(prompt -> {
            if (AgenticIndexMockLlm.isPlanning(prompt)) {
                return "{\"subqueries\":[{\"index\":\""
                    + prefix
                    + "_a\",\"query\":\"sso\"},{\"index\":\""
                    + prefix
                    + "_b\",\"query\":\"sso\"}]}";
            }
            if (AgenticIndexMockLlm.isSufficiency(prompt)) {
                return "{\"sufficient\":true}";
            }
            return "unexpected";
        });

        Response response = TestHelper
            .makeRequest(
                aliasClient,
                "POST",
                "/_plugins/_ml/agentic_index/" + aliasPrefix + "_*/_retrieve",
                ImmutableMap.of(),
                "{\"question\":\"Which plan has SSO?\",\"llm_model_id\":\"" + publicModelId + "\",\"max_rounds\":1}",
                null
            );
        Map<String, Object> result = parseResponseToMap(response);

        List<Map<String, Object>> passages = (List<Map<String, Object>>) result.get("passages");
        // The Pro-only alias filter hides the Free document of <prefix>_a.
        assertEquals(
            List.of(prefix + "_a/1", prefix + "_b/1"),
            passages.stream().map(p -> p.get("index") + "/" + p.get("doc_id")).sorted().toList()
        );
        Map<String, Object> trace = (Map<String, Object>) result.get("trace");
        List<Map<String, Object>> subQueries = (List<Map<String, Object>>) ((List<Map<String, Object>>) trace.get("rounds"))
            .get(0)
            .get("subqueries");
        subQueries.forEach(q -> assertNull(q.toString(), q.get("error")));
        Object warnings = trace.getOrDefault("warnings", List.of());
        assertFalse(warnings.toString(), warnings.toString().contains("statistics unavailable"));
    }

    @SuppressWarnings("unchecked")
    public void testPatternMatchingAliasReachesEachIndexOnce() throws Exception {
        Assume.assumeFalse("Containerized node cannot reach the test-JVM loopback mock LLM", AgenticIndexMockLlm.unreachableFromCluster());
        // An alias whose name matches the pattern, over <prefix>_secret (which the pattern also matches) and <index>.
        String alias = prefix + "_pub";
        adminRequest(
            "POST",
            "/_aliases",
            "{\"actions\":[{\"add\":{\"index\":\""
                + prefix
                + "_secret\",\"alias\":\""
                + alias
                + "\"}},{\"add\":{\"index\":\""
                + index
                + "\",\"alias\":\""
                + alias
                + "\"}}]}"
        );
        String role = "agentic_index_pattern_alias_reader";
        String user = "agentic_index_pattern_alias";
        putRole(
            role,
            "{\"cluster_permissions\": [\""
                + MLAgenticIndexAction.NAME
                + "\", \""
                + MLPredictionTaskAction.NAME
                + "\"], \"index_permissions\": [{\"index_patterns\": [\""
                + alias
                + "\"], \"allowed_actions\": [\"indices:data/read/search\", \"indices:admin/mappings/get\"]}]}"
        );
        RestClient userClient = createUserClient(user, role);
        try {
            String publicModelId = registerPublicMockModel();
            mockLlm.respondWith(prompt -> {
                if (AgenticIndexMockLlm.isPlanning(prompt)) {
                    return "{\"subqueries\":[{\"index\":\""
                        + index
                        + "\",\"query\":\"sso\"},{\"index\":\""
                        + prefix
                        + "_secret\",\"query\":\"falcon\"}]}";
                }
                if (AgenticIndexMockLlm.isSufficiency(prompt)) {
                    return "{\"sufficient\":true}";
                }
                return "unexpected";
            });
            Response response = TestHelper
                .makeRequest(
                    userClient,
                    "POST",
                    "/_plugins/_ml/agentic_index/" + prefix + "_*/_retrieve",
                    ImmutableMap.of(),
                    "{\"question\":\"Which plan has SSO?\",\"llm_model_id\":\"" + publicModelId + "\",\"max_rounds\":1}",
                    null
                );
            Map<String, Object> result = parseResponseToMap(response);
            // The alias grant also authorizes <prefix>_secret by name, so its own check passes and it is searched directly;
            // <index> is reached through the alias. Each is resolved once.
            Map<String, Object> trace = (Map<String, Object>) result.get("trace");
            assertEquals(2, ((Number) trace.get("resolved_index_count")).intValue());
            List<Map<String, Object>> subQueries = (List<Map<String, Object>>) ((List<Map<String, Object>>) trace.get("rounds"))
                .get(0)
                .get("subqueries");
            subQueries.forEach(q -> assertNull(q.toString(), q.get("error")));
            List<Map<String, Object>> passages = (List<Map<String, Object>>) result.get("passages");
            assertEquals(
                List.of(index + "/1", prefix + "_secret/1").stream().sorted().toList(),
                passages.stream().map(p -> p.get("index") + "/" + p.get("doc_id")).sorted().toList()
            );
        } finally {
            userClient.close();
            deleteUser(user);
            securityRequest("DELETE", "/_plugins/_security/api/roles/" + role, "");
        }
    }

    // Requests that fail while resolving the target, so the model is never called.
    private Response retrieve(RestClient client, String target) throws IOException {
        return TestHelper
            .makeRequest(
                client,
                "POST",
                "/_plugins/_ml/agentic_index/" + target + "/_retrieve",
                ImmutableMap.of(),
                "{\"question\":\"q\",\"llm_model_id\":\"m\"}",
                null
            );
    }

    // A deployed mock chat model in a public model group, so any user with the predict permission may use it.
    private String registerPublicMockModel() throws Exception {
        mockLlm = new AgenticIndexMockLlm();
        updateClusterSettings("plugins.ml_commons.trusted_connector_endpoints_regex", List.of("^.*$"));
        updateClusterSettings("plugins.ml_commons.connector.private_ip_enabled", true);
        String groupName = "agentic_index_public_" + randomAlphaOfLength(6).toLowerCase(Locale.ROOT);
        Map<String, Object> group = parseResponseToMap(
            adminRequest("POST", "/_plugins/_ml/model_groups/_register", "{\"name\":\"" + groupName + "\",\"access_mode\":\"public\"}")
        );
        String connectorId = registerConnector(mockLlm.connectorBody());
        Map<String, Object> model = parseResponseToMap(
            adminRequest(
                "POST",
                "/_plugins/_ml/models/_register",
                "{\"name\":\"mock chat\",\"function_name\":\"remote\",\"model_group_id\":\""
                    + group.get("model_group_id")
                    + "\",\"connector_id\":\""
                    + connectorId
                    + "\"}"
            )
        );
        String id = (String) model.get("model_id");
        Response deployed = RestMLRemoteInferenceIT.deployRemoteModel(id);
        waitForTask((String) parseResponseToMap(deployed).get("task_id"), MLTaskState.COMPLETED);
        return id;
    }

    private Response ask(RestClient client) throws IOException {
        return ask(client, index);
    }

    // Asks with the admin's private model, so a user without access to it fails at the first model call.
    private Response ask(RestClient client, String target) throws IOException {
        return TestHelper
            .makeRequest(
                client,
                "POST",
                "/_plugins/_ml/agentic_index/" + target + "/_ask",
                ImmutableMap.of(),
                "{\"question\":\"Which plans include SSO?\",\"llm_model_id\":\"" + modelId + "\"}",
                null
            );
    }

    private RestClient createUserClient(String user, String role) throws IOException {
        String password = SecureMLRestIT.generatePassword(user);
        createUser(user, password, List.of());
        createRoleMapping(role, List.of(user));
        return new SecureRestClientBuilder(getClusterHosts().toArray(new HttpHost[0]), isHttps(), user, password)
            .setSocketTimeout(60000)
            .build();
    }

    private void putRole(String role, String body) throws IOException {
        securityRequest("PUT", "/_plugins/_security/api/roles/" + role, body);
    }

    private Response adminRequest(String method, String endpoint, String body) throws IOException {
        return TestHelper.makeRequest(client(), method, endpoint, ImmutableMap.of(), body, null);
    }

    private Response securityRequest(String method, String endpoint, String body) throws IOException {
        return TestHelper
            .makeRequest(client(), method, endpoint, null, body, ImmutableList.of(new BasicHeader(HttpHeaders.USER_AGENT, "Kibana")));
    }
}
