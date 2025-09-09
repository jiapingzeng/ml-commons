/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.tools;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;

import java.util.HashMap;
import java.util.Map;

import org.junit.Before;
import org.junit.Test;
import org.mockito.Mock;
import org.mockito.MockitoAnnotations;
import org.opensearch.core.action.ActionListener;

public class RemoteListIndexToolTests {

    @Mock
    ActionListener<String> mockedActionListener;

    @Mock
    Map<String, String> mockedParameters;

    private RemoteListIndexTool remoteListIndexTool;

    @Before
    public void setup() {
        MockitoAnnotations.openMocks(this);
        // Create tool with test configuration
        remoteListIndexTool = new RemoteListIndexTool(
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
        assertEquals("RemoteListIndexTool", remoteListIndexTool.getName());
    }

    @Test
    public void testGetType() {
        assertEquals("RemoteListIndexTool", remoteListIndexTool.getType());
    }

    @Test
    public void testGetDescription() {
        assertTrue(remoteListIndexTool.getDescription().contains("OpenSearch cluster"));
        assertTrue(remoteListIndexTool.getDescription().contains("indices"));
    }

    @Test
    public void testValidate_withValidParameters() {
        Map<String, String> parameters = new HashMap<>();
        parameters.put("indices", "[\"index1\", \"index2\"]");
        parameters.put("local", "true");

        assertTrue(remoteListIndexTool.validate(parameters));
    }

    @Test
    public void testValidate_withEmptyParameters() {
        // Should return true since all runtime parameters are optional
        assertTrue(remoteListIndexTool.validate(new HashMap<>()));
    }

    @Test
    public void testValidate_withNullParameters() {
        // Should return true since all runtime parameters are optional
        assertTrue(remoteListIndexTool.validate(null));
    }

    @Test
    public void testValidate_withIndicesOnly() {
        Map<String, String> parameters = new HashMap<>();
        parameters.put("indices", "[\"test-index\"]");

        assertTrue(remoteListIndexTool.validate(parameters));
    }

    @Test
    public void testValidate_withLocalOnly() {
        Map<String, String> parameters = new HashMap<>();
        parameters.put("local", "false");

        assertTrue(remoteListIndexTool.validate(parameters));
    }

    @Test
    public void testRun_withMisconfiguredTool() {
        // This test should expect an IllegalArgumentException during construction
        try {
            RemoteListIndexTool misconfiguredTool = new RemoteListIndexTool(
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
            new RemoteListIndexTool(
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
            new RemoteListIndexTool(
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
    public void testFactory() {
        RemoteListIndexTool.Factory factory = RemoteListIndexTool.Factory.getInstance();

        assertEquals("RemoteListIndexTool", factory.getDefaultType());
        assertTrue(factory.getDefaultDescription().contains("remote OpenSearch cluster"));

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

        RemoteListIndexTool tool = factory.create(config);
        assertEquals("RemoteListIndexTool", tool.getType());
    }

    @Test
    public void testFactory_Singleton() {
        RemoteListIndexTool.Factory factory1 = RemoteListIndexTool.Factory.getInstance();
        RemoteListIndexTool.Factory factory2 = RemoteListIndexTool.Factory.getInstance();

        assertEquals(factory1, factory2);
    }

    @Test
    public void testInputSchema_includesRuntimeParameters() {
        String inputSchema = (String) remoteListIndexTool.getAttributes().get("input_schema");

        // Should include runtime parameters
        assertTrue(inputSchema.contains("indices"));
        assertTrue(inputSchema.contains("local"));

        // Should NOT include endpoint or AWS credentials (configured at creation time)
        assertFalse(inputSchema.contains("endpoint"));
        assertFalse(inputSchema.contains("access_key"));
        assertFalse(inputSchema.contains("secret_key"));
        assertFalse(inputSchema.contains("session_token"));
        assertFalse(inputSchema.contains("region"));
        assertFalse(inputSchema.contains("service_name"));

        // Required parameters should be empty (no required runtime parameters)
        assertTrue(inputSchema.contains("\"required\":") && inputSchema.contains("[]"));
    }

    @Test
    public void testInputSchema_descriptions() {
        String inputSchema = (String) remoteListIndexTool.getAttributes().get("input_schema");

        assertTrue(inputSchema.contains("OpenSearch index name list, separated by comma"));
        assertTrue(inputSchema.contains("Whether to return information from the local node only instead of the cluster manager node"));
    }

    @Test
    public void testDefaultAttributes() {
        RemoteListIndexTool.Factory factory = RemoteListIndexTool.Factory.getInstance();
        Map<String, Object> attributes = factory.getDefaultAttributes();

        assertTrue(attributes.containsKey("input_schema"));
        assertTrue(attributes.containsKey("strict"));
        assertEquals(false, attributes.get("strict"));
    }
}
