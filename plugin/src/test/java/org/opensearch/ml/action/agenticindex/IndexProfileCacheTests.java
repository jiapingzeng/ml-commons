/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.List;
import java.util.Map;

import org.opensearch.Version;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.test.OpenSearchTestCase;

public class IndexProfileCacheTests extends OpenSearchTestCase {

    public void testKeyUsesIndexUuidAndMappingVersion() {
        IndexMetadata index = IndexMetadata
            .builder("docs")
            .settings(settings(Version.CURRENT).put(IndexMetadata.SETTING_INDEX_UUID, "uuid-1"))
            .numberOfShards(1)
            .numberOfReplicas(0)
            .mappingVersion(7)
            .build();
        ClusterState state = ClusterState.builder(new ClusterName("test")).metadata(Metadata.builder().put(index, false)).build();
        ClusterService clusterService = mock(ClusterService.class);
        when(clusterService.state()).thenReturn(state);
        IndexProfileCache cache = new IndexProfileCache(clusterService);

        String key = cache.key("alice", "docs", "docs", List.of("body"));
        assertTrue(key, key.contains("uuid-1/7"));
        assertNotEquals(key, cache.key("bob", "docs", "docs", List.of("body")));
        assertNotEquals(key, cache.key("alice", "docs", "docs", List.of("title")));
        // Two filtered aliases on one index can see different documents.
        assertNotEquals(cache.key("alice", "alias-a", "docs", List.of()), cache.key("alice", "alias-b", "docs", List.of()));
        assertNull(cache.key("alice", "missing", "missing", List.of()));
    }

    public void testGetAndPut() {
        IndexProfileCache cache = new IndexProfileCache(index -> "v1", TimeValue.timeValueMinutes(1));
        IndexProfile profile = IndexProfile.fromMappingProperties("docs", Map.of("body", Map.of("type", "text")));
        String key = cache.key(null, "docs", "docs", List.of());
        assertNull(cache.get(key));
        cache.put(key, profile);
        assertSame(profile, cache.get(key));
        // Unknown index versions are never cached.
        cache.put(null, profile);
        assertNull(cache.get(null));
    }
}
