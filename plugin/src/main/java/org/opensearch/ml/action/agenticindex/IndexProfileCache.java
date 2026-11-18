/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import java.util.List;
import java.util.function.Function;

import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.service.ClusterService;
import org.opensearch.common.cache.Cache;
import org.opensearch.common.cache.CacheBuilder;
import org.opensearch.common.unit.TimeValue;

/**
 * Short-lived cache of index profiles, so repeated questions over one index skip the statistics search. Entries are
 * keyed by caller, requested index or alias, index UUID, mapping version and requested text fields: a mapping change, a
 * recreated index, a different filtered alias or a different user (whose document-level security may differ) never
 * reuses an entry.
 */
public class IndexProfileCache {

    static final TimeValue TTL = TimeValue.timeValueMinutes(1);
    static final int MAX_ENTRIES = 256;

    private final Function<String, String> indexVersion;
    private final Cache<String, IndexProfile> cache;

    public IndexProfileCache(ClusterService clusterService) {
        this(index -> {
            IndexMetadata metadata = clusterService.state().metadata().index(index);
            return metadata == null ? null : metadata.getIndexUUID() + "/" + metadata.getMappingVersion();
        }, TTL);
    }

    /**
     * @param indexVersion maps a concrete index name to a string that changes with the index UUID and mapping, or null
     *                     when the index is unknown (no caching)
     */
    IndexProfileCache(Function<String, String> indexVersion, TimeValue ttl) {
        this.indexVersion = indexVersion;
        this.cache = CacheBuilder.<String, IndexProfile>builder().setExpireAfterWrite(ttl).setMaximumWeight(MAX_ENTRIES).build();
    }

    /**
     * Cache key for a profile, or null when the index version is unknown. The requested target is part of the key
     * because two filtered aliases on one index see different documents.
     */
    public String key(String caller, String target, String concreteIndex, List<String> textFields) {
        String version = indexVersion.apply(concreteIndex);
        return version == null
            ? null
            : String.join("\u0000", String.valueOf(caller), target, concreteIndex, version, String.join(",", textFields));
    }

    public IndexProfile get(String key) {
        return key == null ? null : cache.get(key);
    }

    public void put(String key, IndexProfile profile) {
        if (key != null) {
            cache.put(key, profile);
        }
    }
}
