/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;

import java.util.HashMap;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.core.action.ActionListener;

public class RemoteSearchIndexToolTests {

    @Mock
    ActionListener<String> mockedActionListener;

    private RemoteSearchIndexTool remoteSearchIndexTool;

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        // Create tool with test configuration
        remoteSearchIndexTool = new RemoteSearchIndexTool(
            "https://search-domain.us-east-1.es.amazonaws.com",
            "AKIAIOSFODNN7EXAMPLE",
            "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "AQoEXAMPLEH4aoAH0gNCAPyJxz4BlCFFxWNE1OPTgk5TthT+FvwqnKwRcOIfrRh3c/LTo6UDdyJwOOvEVPvLXCrrrUtdnniCEXAMPLE/IvU1dYUg2RVAJBanLiHb4IgRmpRV3zrkuWJOgQs8IZZaIv2BXIa2R4Olgk",
            "us-east-1",
            "es"
        );
    }

    @Test
    public void testGetName() {
        assertEquals("RemoteSearchIndexTool", remoteSearchIndexTool.getName());
    }

    @Test
    public void testGetType() {
        assertEquals("RemoteSearchIndexTool", remoteSearchIndexTool.getType());
    }

    @Test
    public void testGetDescription() {
        assertTrue(remoteSearchIndexTool.getDescription().contains("OpenSearch"));
        assertTrue(remoteSearchIndexTool.getDescription().contains("DSL"));
    }

    @Test
    public void testValidate_withValidDirectParameters() {
        Map<String, String> parameters = new HashMap<>();
        parameters.put("index", "test-index");
        parameters.put("query", "{\"query\":{\"match_all\":{}}}");

        assertTrue(remoteSearchIndexTool.validate(parameters));
    }

    @Test
    public void testValidate_withValidInputParameter() {
        Map<String, String> parameters = new HashMap<>();
        parameters.put("input", "{\"index\":\"test-index\",\"query\":{\"query\":{\"match_all\":{}}}");

        assertTrue(remoteSearchIndexTool.validate(parameters));
    }

    @Test
    public void testValidate_withoutIndex() {
        Map<String, String> parameters = new HashMap<>();
        parameters.put("query", "{\"query\":{\"match_all\":{}}}");

        assertFalse(remoteSearchIndexTool.validate(parameters));
    }

    @Test
    public void testValidate_withoutQuery() {
        Map<String, String> parameters = new HashMap<>();
        parameters.put("index", "test-index");

        assertFalse(remoteSearchIndexTool.validate(parameters));
    }

    @Test
    public void testValidate_withMisconfiguredTool() {
        // This test should expect an IllegalArgumentException during construction
        try {
            RemoteSearchIndexTool misconfiguredTool = new RemoteSearchIndexTool(
                null,  // missing endpoint
                "AKIAIOSFODNN7EXAMPLE",
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "AQoEXAMPLEH4aoAH0gNCAPyJxz4BlCFFxWNE1OPTgk5TthT+FvwqnKwRcOIfrRh3c/LTo6UDdyJwOOvEVPvLXCrrrUtdnniCEXAMPLE/IvU1dYUg2RVAJBanLiHb4IgRmpRV3zrkuWJOgQs8IZZaIv2BXIa2R4Olgk",
                "us-east-1",
                "es"
            );
            // Should not reach here
            assertTrue("Expected IllegalArgumentException during construction", false);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("endpoint is required"));
        }
    }

    @Test
    public void testValidate_withoutIndexAndQuery() {
        Map<String, String> parameters = new HashMap<>();
        // No index or query provided

        assertFalse(remoteSearchIndexTool.validate(parameters));
    }

    @Test
    public void testValidate_withNullParameters() {
        assertFalse(remoteSearchIndexTool.validate(null));
    }

    @Test
    public void testValidate_withEmptyParameters() {
        assertFalse(remoteSearchIndexTool.validate(new HashMap<>()));
    }

    @Test
    public void testRun_withMisconfiguredTool() {
        // This test should expect an IllegalArgumentException during construction
        try {
            RemoteSearchIndexTool misconfiguredTool = new RemoteSearchIndexTool(
                null,  // missing endpoint
                "AKIAIOSFODNN7EXAMPLE",
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "AQoEXAMPLEH4aoAH0gNCAPyJxz4BlCFFxWNE1OPTgk5TthT+FvwqnKwRcOIfrRh3c/LTo6UDdyJwOOvEVPvLXCrrrUtdnniCEXAMPLE/IvU1dYUg2RVAJBanLiHb4IgRmpRV3zrkuWJOgQs8IZZaIv2BXIa2R4Olgk",
                "us-east-1",
                "es"
            );
            // Should not reach here
            assertTrue("Expected IllegalArgumentException during construction", false);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("endpoint is required"));
        }
    }

    @Test
    public void testConstructor_withMissingEndpoint() {
        try {
            new RemoteSearchIndexTool(
                null,  // missing endpoint
                "AKIAIOSFODNN7EXAMPLE",
                "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                "AQoEXAMPLEH4aoAH0gNCAPyJxz4BlCFFxWNE1OPTgk5TthT+FvwqnKwRcOIfrRh3c/LTo6UDdyJwOOvEVPvLXCrrrUtdnniCEXAMPLE/IvU1dYUg2RVAJBanLiHb4IgRmpRV3zrkuWJOgQs8IZZaIv2BXIa2R4Olgk",
                "us-east-1",
                "es"
            );
            // Should not reach here
            assertTrue("Expected IllegalArgumentException", false);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("endpoint is required"));
        }
    }

    @Test
    public void testConstructor_withMissingAwsCredentials() {
        try {
            new RemoteSearchIndexTool(
                "https://remote-cluster:9200",
                "AKIAIOSFODNN7EXAMPLE",
                null,  // missing secret key
                "AQoEXAMPLEH4aoAH0gNCAPyJxz4BlCFFxWNE1OPTgk5TthT+FvwqnKwRcOIfrRh3c/LTo6UDdyJwOOvEVPvLXCrrrUtdnniCEXAMPLE/IvU1dYUg2RVAJBanLiHb4IgRmpRV3zrkuWJOgQs8IZZaIv2BXIa2R4Olgk",
                "us-east-1",
                "es"
            );
            // Should not reach here
            assertTrue("Expected IllegalArgumentException", false);
        } catch (IllegalArgumentException e) {
            assertTrue(e.getMessage().contains("AWS credentials"));
        }
    }

    @Test
    public void testRun_withMissingIndex() {
        Map<String, String> parameters = new HashMap<>();
        parameters.put("query", "{\"query\":{\"match_all\":{}}}");

        doAnswer(invocation -> {
            Exception exception = invocation.getArgument(0);
            assertTrue(exception instanceof IllegalArgumentException);
            assertTrue(exception.getMessage().contains("requires both 'index' and 'query' parameters"));
            return null;
        }).when(mockedActionListener).onFailure(any(Exception.class));

        remoteSearchIndexTool.run(parameters, mockedActionListener);
    }

    @Test
    public void testRun_withMissingQuery() {
        Map<String, String> parameters = new HashMap<>();
        parameters.put("index", "test-index");

        doAnswer(invocation -> {
            Exception exception = invocation.getArgument(0);
            assertTrue(exception instanceof IllegalArgumentException);
            assertTrue(exception.getMessage().contains("requires both 'index' and 'query' parameters"));
            return null;
        }).when(mockedActionListener).onFailure(any(Exception.class));

        remoteSearchIndexTool.run(parameters, mockedActionListener);
    }

    @Test
    public void testRun_withInvalidJsonInput() {
        Map<String, String> parameters = new HashMap<>();
        parameters.put("input", "{invalid json}");

        doAnswer(invocation -> {
            Exception exception = invocation.getArgument(0);
            assertTrue(exception instanceof IllegalArgumentException);
            assertTrue(exception.getMessage().contains("Invalid JSON input"));
            return null;
        }).when(mockedActionListener).onFailure(any(Exception.class));

        remoteSearchIndexTool.run(parameters, mockedActionListener);
    }

    @Test
    public void testFactory() {
        RemoteSearchIndexTool.Factory factory = RemoteSearchIndexTool.Factory.getInstance();

        assertEquals("RemoteSearchIndexTool", factory.getDefaultType());
        assertTrue(factory.getDefaultDescription().contains("OpenSearch"));

        Map<String, Object> config = new HashMap<>();
        config.put("endpoint", "https://search-domain.us-east-1.es.amazonaws.com");
        config.put("access_key", "AKIAIOSFODNN7EXAMPLE");
        config.put("secret_key", "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY");
        config
            .put(
                "session_token",
                "AQoEXAMPLEH4aoAH0gNCAPyJxz4BlCFFxWNE1OPTgk5TthT+FvwqnKwRcOIfrRh3c/LTo6UDdyJwOOvEVPvLXCrrrUtdnniCEXAMPLE/IvU1dYUg2RVAJBanLiHb4IgRmpRV3zrkuWJOgQs8IZZaIv2BXIa2R4Olgk"
            );
        config.put("region", "us-east-1");
        config.put("service_name", "es");

        RemoteSearchIndexTool tool = factory.create(config);
        assertEquals("RemoteSearchIndexTool", tool.getType());
    }

    @Test
    public void testFactory_Singleton() {
        RemoteSearchIndexTool.Factory factory1 = RemoteSearchIndexTool.Factory.getInstance();
        RemoteSearchIndexTool.Factory factory2 = RemoteSearchIndexTool.Factory.getInstance();

        assertEquals(factory1, factory2);
    }

    @Test
    public void testInputSchema_includesSearchParameters() {
        String inputSchema = (String) remoteSearchIndexTool.getAttributes().get("input_schema");

        // Should include search parameters
        assertTrue(inputSchema.contains("index"));
        assertTrue(inputSchema.contains("query"));
        assertTrue(inputSchema.contains("return_raw_response"));

        // Should NOT include endpoint or AWS credentials (configured at creation time)
        assertFalse(inputSchema.contains("endpoint"));
        assertFalse(inputSchema.contains("access_key"));
        assertFalse(inputSchema.contains("secret_key"));
        assertFalse(inputSchema.contains("session_token"));
        assertFalse(inputSchema.contains("region"));
        assertFalse(inputSchema.contains("service_name"));

        // Query should be required
        assertTrue(inputSchema.contains("\"required\":") && inputSchema.contains("\"query\""));
    }

    @Test
    public void testInputSchema_descriptions() {
        String inputSchema = (String) remoteSearchIndexTool.getAttributes().get("input_schema");

        assertTrue(inputSchema.contains("OpenSearch index name to search"));
        assertTrue(inputSchema.contains("OpenSearch search index query in DSL format"));
        assertTrue(inputSchema.contains("Alternative input format containing index and query as JSON string"));
        assertTrue(inputSchema.contains("Whether to return the full search response"));
    }

    @Test
    public void testDefaultAttributes() {
        RemoteSearchIndexTool.Factory factory = RemoteSearchIndexTool.Factory.getInstance();
        Map<String, Object> attributes = factory.getDefaultAttributes();

        assertTrue(attributes.containsKey("input_schema"));
        assertTrue(attributes.containsKey("strict"));
        assertEquals(false, attributes.get("strict"));
    }
}
