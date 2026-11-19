/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import static org.mockito.ArgumentMatchers.any;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;

import org.opensearch.OpenSearchSecurityException;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.Version;
import org.opensearch.action.admin.indices.mapping.get.GetMappingsRequest;
import org.opensearch.action.admin.indices.mapping.get.GetMappingsResponse;
import org.opensearch.cluster.ClusterName;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.AliasMetadata;
import org.opensearch.cluster.metadata.DataStream;
import org.opensearch.cluster.metadata.IndexAbstraction;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.cluster.metadata.Metadata;
import org.opensearch.common.regex.Regex;
import org.opensearch.common.settings.Settings;
import org.opensearch.common.util.concurrent.ThreadContext;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.tasks.TaskCancelledException;
import org.opensearch.core.tasks.TaskId;
import org.opensearch.index.IndexNotFoundException;
import org.opensearch.ml.action.agenticindex.IndexResolver.ResolvedIndex;
import org.opensearch.test.OpenSearchTestCase;
import org.opensearch.transport.client.AdminClient;
import org.opensearch.transport.client.Client;
import org.opensearch.transport.client.IndicesAdminClient;

public class IndexResolverTests extends OpenSearchTestCase {

    private static final TaskId PARENT = new TaskId("node-1", 7L);

    private ClusterState state;
    // Indices and aliases the caller may read; null means every index.
    private Set<String> readable;
    private final List<GetMappingsRequest> mappingRequests = new ArrayList<>();
    // Mapping _meta of an index, by index name.
    private final Map<String, Object> metaByIndex = new HashMap<>();
    private IndexResolver resolver;
    private final AtomicBoolean cancelled = new AtomicBoolean();
    private final AtomicBoolean expired = new AtomicBoolean();

    @Override
    public void setUp() throws Exception {
        super.setUp();
        Client client = mock(Client.class);
        AdminClient admin = mock(AdminClient.class);
        IndicesAdminClient indices = mock(IndicesAdminClient.class);
        when(client.admin()).thenReturn(admin);
        when(admin.indices()).thenReturn(indices);
        doAnswer(invocation -> {
            GetMappingsRequest request = invocation.getArgument(0);
            ActionListener<GetMappingsResponse> listener = invocation.getArgument(1);
            mappingRequests.add(request);
            try {
                GetMappingsResponse response = mock(GetMappingsResponse.class);
                Map<String, MappingMetadata> mappings = mappingsAsCaller(request.indices());
                when(response.getMappings()).thenReturn(mappings);
                listener.onResponse(response);
            } catch (Exception e) {
                listener.onFailure(e);
            }
            return null;
        }).when(indices).getMappings(any(GetMappingsRequest.class), any());
        resolver = new IndexResolver(client, () -> state, new IndexNameExpressionResolver(new ThreadContext(Settings.EMPTY)));
        state = state(index("logs-a"), index("logs-b"), index("logs-secret"));
    }

    // Mimics the security plugin without do_not_fail_on_forbidden: one unreadable name or match rejects the request.
    // A pattern matches index, alias and data stream names; an index reached through a readable alias is readable.
    private Map<String, MappingMetadata> mappingsAsCaller(String[] names) {
        Map<String, MappingMetadata> result = new TreeMap<>();
        for (String name : names) {
            Map<String, Boolean> matched = new TreeMap<>();
            if (Regex.isSimpleMatchPattern(name)) {
                for (IndexAbstraction abstraction : state.metadata().getIndicesLookup().values()) {
                    if (!Regex.simpleMatch(name, abstraction.getName()) || abstraction.isHidden()) {
                        continue;
                    }
                    boolean viaReadable = abstraction.getType() != IndexAbstraction.Type.CONCRETE_INDEX
                        && readable != null
                        && readable.contains(abstraction.getName());
                    abstraction.getIndices().forEach(m -> matched.merge(m.getIndex().getName(), viaReadable, Boolean::logicalOr));
                }
            } else {
                if (readable != null && !readable.contains(name) && state.metadata().getIndicesLookup().get(name) == null) {
                    throw forbidden();
                }
                IndexAbstraction abstraction = state.metadata().getIndicesLookup().get(name);
                if (abstraction == null) {
                    throw new IndexNotFoundException(name);
                }
                boolean viaReadable = readable != null && readable.contains(name);
                abstraction.getIndices().forEach(m -> matched.put(m.getIndex().getName(), viaReadable));
            }
            matched.forEach((index, viaReadable) -> {
                if (readable != null && !readable.contains(index) && !viaReadable) {
                    throw forbidden();
                }
                Map<String, Object> mapping = new HashMap<>(Map.of("properties", Map.of("t", Map.of("type", "text"))));
                if (metaByIndex.containsKey(index)) {
                    mapping.put("_meta", metaByIndex.get(index));
                }
                result.put(index, new MappingMetadata(index, mapping));
            });
        }
        return result;
    }

    private static OpenSearchSecurityException forbidden() {
        return new OpenSearchSecurityException("no permissions for [indices:admin/mappings/get]", RestStatus.FORBIDDEN);
    }

    private static IndexMetadata.Builder index(String name) {
        return index(name, false);
    }

    private static IndexMetadata.Builder hidden(String name) {
        return index(name, true);
    }

    private static IndexMetadata.Builder index(String name, boolean hidden) {
        return IndexMetadata
            .builder(name)
            .settings(
                Settings
                    .builder()
                    .put(IndexMetadata.SETTING_VERSION_CREATED, Version.CURRENT)
                    .put(IndexMetadata.SETTING_NUMBER_OF_SHARDS, 1)
                    .put(IndexMetadata.SETTING_NUMBER_OF_REPLICAS, 0)
                    .put(IndexMetadata.SETTING_INDEX_HIDDEN, hidden)
            );
    }

    private static ClusterState state(IndexMetadata.Builder... indices) {
        Metadata.Builder metadata = Metadata.builder();
        for (IndexMetadata.Builder index : indices) {
            metadata.put(index);
        }
        return ClusterState.builder(new ClusterName("test")).metadata(metadata).build();
    }

    private List<ResolvedIndex> resolve(String expression, int max) {
        AtomicReference<List<ResolvedIndex>> result = new AtomicReference<>();
        AtomicReference<Exception> failure = new AtomicReference<>();
        resolver.resolve(expression, max, PARENT, cancelled::get, expired::get, ActionListener.wrap(result::set, failure::set));
        if (failure.get() != null) {
            throw new AssertionError(failure.get());
        }
        return result.get();
    }

    private Exception resolveFailure(String expression, int max) {
        AtomicReference<Exception> failure = new AtomicReference<>();
        resolver
            .resolve(
                expression,
                max,
                PARENT,
                cancelled::get,
                expired::get,
                ActionListener.wrap(r -> fail("expected failure, got " + r), failure::set)
            );
        assertNotNull(failure.get());
        return failure.get();
    }

    private static List<String> names(List<ResolvedIndex> resolved) {
        return resolved.stream().map(ResolvedIndex::name).toList();
    }

    public void testSingleIndexKeepsRequestedTarget() {
        List<ResolvedIndex> resolved = resolve("logs-a", 500);
        assertEquals(List.of(new ResolvedIndex("logs-a", "logs-a", false, Map.of("t", Map.of("type", "text")), null)), resolved);
        assertEquals(1, mappingRequests.size());
        assertEquals(PARENT, mappingRequests.get(0).getParentTask());
    }

    public void testPatternResolvingToOneIndexKeepsPattern() {
        List<ResolvedIndex> resolved = resolve("logs-se*", 500);
        assertEquals("logs-secret", resolved.get(0).name());
        assertEquals("logs-se*", resolved.get(0).target());
    }

    public void testCommaListAndPatternResolveToSortedIndices() {
        List<ResolvedIndex> resolved = resolve("logs-b, logs-*", 500);
        assertEquals(List.of("logs-a", "logs-b", "logs-secret"), names(resolved));
        resolved.forEach(r -> assertEquals(r.name(), r.target()));
        assertEquals(List.of("logs-b", "logs-*"), List.of(mappingRequests.get(0).indices()));
    }

    public void testPatternsSkipHiddenSystemAndDotIndicesUnlessNamed() {
        state = state(index("logs-a"), hidden("logs-hidden"), index(".logs-internal"), index("logs-system").system(true));
        assertEquals(List.of("logs-a"), names(resolve("*logs*", 500)));
        assertEquals(List.of(".logs-internal", "logs-a", "logs-system"), names(resolve("logs-*,.logs-internal,logs-system", 500)));
    }

    public void testCapRejectsTooManyIndices() {
        OpenSearchStatusException e = (OpenSearchStatusException) resolveFailure("logs-*", 2);
        assertEquals(RestStatus.BAD_REQUEST, e.status());
        assertEquals(
            "[logs-*] resolves to more than 2 indices that you can read, the limit set by plugins.ml_commons.agentic_index_max_indices. "
                + "Use a narrower target.",
            e.getMessage()
        );
        assertEquals(3, resolve("logs-*", 3).size());
    }

    public void testPatternWithPartialAccessKeepsReadableIndicesOnly() {
        readable = Set.of("logs-a", "logs-b");
        List<ResolvedIndex> resolved = resolve("logs-*", 2);
        assertEquals(List.of("logs-a", "logs-b"), names(resolved));
        // The pattern request was rejected, then each match was checked as the caller.
        assertEquals(List.of("logs-*"), List.of(mappingRequests.get(0).indices()));
        assertEquals(4, mappingRequests.size());
        mappingRequests.forEach(r -> assertEquals(PARENT, r.getParentTask()));
    }

    public void testCapCountsOnlyReadableIndices() {
        state = state(index("logs-a"), index("logs-b"), index("logs-c"), index("logs-secret"));
        readable = Set.of("logs-a", "logs-b", "logs-c");
        OpenSearchStatusException e = (OpenSearchStatusException) resolveFailure("logs-*", 2);
        assertEquals(RestStatus.BAD_REQUEST, e.status());
        assertTrue(e.getMessage(), e.getMessage().startsWith("[logs-*] resolves to more than 2 indices that you can read"));
    }

    public void testPerIndexChecksAreCapped() {
        readable = Set.of("logs-a");
        OpenSearchStatusException e = (OpenSearchStatusException) resolveFailure("logs-*", 1);
        assertEquals(RestStatus.BAD_REQUEST, e.status());
        assertEquals(
            "[logs-*] matches indices that you cannot read, and more than 2 indices to check one by one (2 times "
                + "plugins.ml_commons.agentic_index_max_indices). Use a narrower target that matches only indices you can read.",
            e.getMessage()
        );
        // Only the rejected pattern request ran.
        assertEquals(1, mappingRequests.size());
    }

    public void testDeadlineDuringPerIndexChecksTimesOut() {
        readable = Set.of("logs-a");
        expired.set(true);
        OpenSearchStatusException e = (OpenSearchStatusException) resolveFailure("logs-*", 500);
        assertTrue(e instanceof IndexResolver.TimedOutException);
        assertEquals(RestStatus.REQUEST_TIMEOUT, e.status());
        assertTrue(e.getMessage(), e.getMessage().startsWith("The request timed out while checking"));
        assertEquals(1, mappingRequests.size());
    }

    public void testPatternMatchingAliasNamesSearchesThroughTheAlias() {
        state = state(
            index("docs-1").putAlias(AliasMetadata.builder("pub-a").filter("{\"term\":{\"t\":\"x\"}}")),
            index("docs-2").putAlias(AliasMetadata.builder("pub-b")),
            index("pub-direct")
        );
        List<ResolvedIndex> resolved = resolve("pub-*", 500);
        assertEquals(List.of("docs-1", "docs-2", "pub-direct"), names(resolved));
        assertEquals(new ResolvedIndex("docs-1", "pub-a", true, Map.of("t", Map.of("type", "text")), null), resolved.get(0));
        assertEquals("pub-b", resolved.get(1).target());
        assertTrue(resolved.get(1).filterByIndex());
        assertEquals("pub-direct", resolved.get(2).target());
        assertFalse(resolved.get(2).filterByIndex());
    }

    public void testAliasOnlyReaderWithPatternFallbackChecksThroughAlias() {
        state = state(
            index("docs-1").putAlias(AliasMetadata.builder("pub-a")),
            index("docs-2").putAlias(AliasMetadata.builder("pub-b")),
            index("pub-direct")
        );
        // Readable only through the pub-a alias, not by index name; pub-b is unreadable.
        readable = Set.of("pub-a", "pub-direct");
        List<ResolvedIndex> resolved = resolve("pub-*", 500);
        assertEquals(List.of("docs-1", "pub-direct"), names(resolved));
        assertEquals("pub-a", resolved.get(0).target());
        assertTrue(resolved.get(0).filterByIndex());
        assertEquals("pub-direct", resolved.get(1).target());
        List<String> checked = mappingRequests.subList(1, mappingRequests.size()).stream().map(r -> r.indices()[0]).sorted().toList();
        assertEquals(List.of("pub-a", "pub-b", "pub-direct"), checked);
    }

    public void testAliasCheckDoesNotCreditIndexMatchedByName() {
        state = state(
            index("pub-1").putAlias(AliasMetadata.builder("pub-a")),
            index("raw-2").putAlias(AliasMetadata.builder("pub-a")),
            index("pub-z")
        );
        // Readable only through alias pub-a, which here does not authorize pub-1 by name; pub-1's own check fails, so it is
        // not searched directly.
        readable = Set.of("pub-a");
        List<ResolvedIndex> resolved = resolve("pub-*", 500);
        assertEquals(List.of("raw-2"), names(resolved));
        assertEquals("pub-a", resolved.get(0).target());
        assertTrue(resolved.get(0).filterByIndex());
    }

    public void testAliasCheckCountsEachIndexOnce() {
        state = state(
            index("pub-1").putAlias(AliasMetadata.builder("pub-a")),
            index("raw-2").putAlias(AliasMetadata.builder("pub-a")),
            index("pub-2"),
            index("pub-3"),
            index("pub-q"),
            index("pub-z")
        );
        // Five readable indices: pub-1, pub-2, pub-3, raw-2 (through pub-a) and pub-z, checked last.
        readable = Set.of("pub-1", "pub-2", "pub-3", "pub-a", "pub-z");
        OpenSearchStatusException e = (OpenSearchStatusException) resolveFailure("pub-*", 4);
        assertEquals(RestStatus.BAD_REQUEST, e.status());
        assertTrue(e.getMessage(), e.getMessage().startsWith("[pub-*] resolves to more than 4 indices that you can read"));
        List<String> checked = mappingRequests.subList(1, mappingRequests.size()).stream().map(r -> r.indices()[0]).toList();
        assertTrue(checked.toString(), checked.contains("pub-z"));
        assertEquals(5, resolve("pub-*", 5).size());
    }

    public void testDescriptionComesFromMappingMeta() {
        metaByIndex.put("logs-a", Map.of("description", "Application logs"));
        metaByIndex.put("logs-b", Map.of("description", 42));
        List<ResolvedIndex> resolved = resolve("logs-*", 500);
        assertEquals("Application logs", resolved.get(0).description());
        assertNull(resolved.get(1).description());
        assertNull(resolved.get(2).description());
        assertEquals("Application logs", resolve("logs-a", 500).get(0).description());
    }

    public void testDataStreamMatchedByPatternIncludesBackingIndices() {
        IndexMetadata backing = hidden(DataStream.getDefaultBackingIndexName("logs-app", 1)).build();
        Metadata metadata = Metadata
            .builder()
            .put(backing, false)
            .put(index("logs-a"))
            .put(new DataStream("logs-app", new DataStream.TimestampField("@timestamp"), List.of(backing.getIndex())))
            .build();
        state = ClusterState.builder(new ClusterName("test")).metadata(metadata).build();
        List<ResolvedIndex> resolved = resolve("logs-*", 500);
        assertEquals(List.of(backing.getIndex().getName(), "logs-a"), names(resolved));
        assertEquals("logs-app", resolved.get(0).target());
        assertTrue(resolved.get(0).filterByIndex());
        assertEquals("logs-a", resolved.get(1).target());
    }

    public void testUnreadableNamedIndexIsForbidden() {
        readable = Set.of("logs-a");
        assertEquals(RestStatus.FORBIDDEN, ((OpenSearchSecurityException) resolveFailure("logs-a,logs-secret", 500)).status());
        assertEquals(RestStatus.FORBIDDEN, ((OpenSearchSecurityException) resolveFailure("logs-*,logs-secret", 500)).status());
        // A missing name looks the same as an unreadable one.
        assertEquals(RestStatus.FORBIDDEN, ((OpenSearchSecurityException) resolveFailure("logs-*,no-such-index", 500)).status());
    }

    public void testNoReadableMatchIsNotFound() {
        readable = Set.of();
        OpenSearchStatusException e = (OpenSearchStatusException) resolveFailure("logs-*", 500);
        assertEquals(RestStatus.NOT_FOUND, e.status());
        assertEquals("No indices that you can read match [logs-*]", e.getMessage());
        readable = null;
        assertEquals("No indices that you can read match [zzz-*]", resolveFailure("zzz-*", 500).getMessage());
    }

    public void testUnknownIndexIsNotFound() {
        assertTrue(resolveFailure("no-such-index", 500) instanceof IndexNotFoundException);
    }

    public void testNamedAliasIsSearchedThroughAliasAndNarrowed() {
        state = state(
            index("docs-1").putAlias(AliasMetadata.builder("pub")),
            index("docs-2").putAlias(AliasMetadata.builder("pub")),
            index("other")
        );
        List<ResolvedIndex> resolved = resolve("pub,other", 500);
        assertEquals(List.of("docs-1", "docs-2", "other"), names(resolved));
        assertEquals("pub", resolved.get(0).target());
        assertTrue(resolved.get(0).filterByIndex());
        assertEquals("other", resolved.get(2).target());
        assertFalse(resolved.get(2).filterByIndex());
        // A pattern that matches the index name reaches it directly.
        resolved = resolve("pub,docs-2*", 500);
        assertEquals("pub", resolved.get(0).target());
        assertEquals("docs-2", resolved.get(1).target());
    }

    public void testReadableAliasWithPatternFallback() {
        state = state(index("docs-1").putAlias(AliasMetadata.builder("pub")), index("logs-a"), index("logs-secret"));
        readable = Set.of("pub", "logs-a");
        List<ResolvedIndex> resolved = resolve("pub,logs-*", 500);
        assertEquals(List.of("docs-1", "logs-a"), names(resolved));
        assertEquals("pub", resolved.get(0).target());
    }

    public void testCancelledDuringPerIndexChecks() {
        readable = Set.of("logs-a");
        cancelled.set(true);
        assertTrue(resolveFailure("logs-*", 500) instanceof TaskCancelledException);
        // Only the rejected pattern request ran.
        assertEquals(1, mappingRequests.size());
    }
}
