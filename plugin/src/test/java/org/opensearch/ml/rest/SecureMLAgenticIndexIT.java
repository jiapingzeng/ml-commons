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

import org.apache.hc.core5.http.HttpHeaders;
import org.apache.hc.core5.http.HttpHost;
import org.apache.hc.core5.http.message.BasicHeader;
import org.junit.After;
import org.junit.Before;
import org.opensearch.client.Response;
import org.opensearch.client.ResponseException;
import org.opensearch.client.RestClient;
import org.opensearch.commons.rest.SecureRestClientBuilder;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexAction;
import org.opensearch.ml.common.transport.prediction.MLPredictionTaskAction;
import org.opensearch.ml.utils.TestHelper;

import com.google.common.collect.ImmutableList;
import com.google.common.collect.ImmutableMap;

/**
 * Agentic index authorization with the security plugin: every step runs as the caller, so the caller needs read access
 * to the index and access to the LLM's model group in addition to the agentic index cluster permission.
 */
public class SecureMLAgenticIndexIT extends MLCommonsRestTestCase {

    private static final String NO_READ_ROLE = "agentic_index_no_index_read";
    private static final String READER_ROLE = "agentic_index_reader";

    private final String noReadUser = "agentic_index_no_read";
    private final String readerUser = "agentic_index_reader";
    private RestClient noReadClient;
    private RestClient readerClient;
    private String index;
    private String modelId;

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
        deleteUser(noReadUser);
        deleteUser(readerUser);
        securityRequest("DELETE", "/_plugins/_security/api/roles/" + NO_READ_ROLE, "");
        securityRequest("DELETE", "/_plugins/_security/api/roles/" + READER_ROLE, "");
        deleteIndexWithAdminClient(index);
        updateClusterSettings(ML_COMMONS_AGENTIC_INDEX_ENABLED.getKey(), null);
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

    private Response ask(RestClient client) throws IOException {
        return TestHelper
            .makeRequest(
                client,
                "POST",
                "/_plugins/_ml/agentic_index/" + index + "/_ask",
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
