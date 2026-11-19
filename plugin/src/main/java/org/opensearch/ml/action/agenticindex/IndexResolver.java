/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import static org.opensearch.ml.common.settings.MLCommonsSettings.ML_COMMONS_AGENTIC_INDEX_MAX_INDICES;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import java.util.SortedMap;
import java.util.TreeMap;
import java.util.TreeSet;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.BooleanSupplier;
import java.util.function.Supplier;

import org.opensearch.ExceptionsHelper;
import org.opensearch.OpenSearchStatusException;
import org.opensearch.action.admin.indices.mapping.get.GetMappingsRequest;
import org.opensearch.action.support.IndicesOptions;
import org.opensearch.cluster.ClusterState;
import org.opensearch.cluster.metadata.IndexAbstraction;
import org.opensearch.cluster.metadata.IndexMetadata;
import org.opensearch.cluster.metadata.IndexNameExpressionResolver;
import org.opensearch.cluster.metadata.MappingMetadata;
import org.opensearch.common.regex.Regex;
import org.opensearch.core.action.ActionListener;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.core.tasks.TaskCancelledException;
import org.opensearch.core.tasks.TaskId;
import org.opensearch.ml.action.agenticindex.BoundedFanOut.Outcome;
import org.opensearch.transport.client.Client;

/**
 * Resolves an agentic index target (index, alias, pattern or comma-separated list) to the concrete indices the caller
 * may read, with their mappings. Every mapping read runs as the caller. When the caller cannot read every index a
 * pattern matches (the security plugin then rejects the whole request), each matching index is checked on its own and
 * only the readable ones are kept, so neither the result nor an error reveals indices the caller cannot read. That check
 * covers at most {@link #CHECK_LIMIT_FACTOR} times the index cap and stops at the request deadline.
 * Pattern matches exclude hidden, system and dot-prefixed indices, except backing indices of a matched data stream; an
 * explicitly named index or alias is kept. An index reached only through an alias or data stream (named, or matched by
 * a pattern) is searched through it.
 */
public class IndexResolver {

    static final int CHECK_CONCURRENCY = 16;
    static final int CHECK_LIMIT_FACTOR = 2;

    /**
     * One resolved index. Searches go to {@code target}: the index itself, the alias or data stream it was reached
     * through (so alias filters and alias-only permissions still apply, with {@code filterByIndex} limiting the search
     * to this index), or the original target when that resolves to this index alone. {@code description} is the
     * mapping's {@code _meta.description} when it is a string, otherwise null.
     */
    public record ResolvedIndex(String name, String target, boolean filterByIndex, Map<String, Object> properties, String description) {
    }

    private final Client client;
    private final Supplier<ClusterState> clusterState;
    private final IndexNameExpressionResolver expressionResolver;

    public IndexResolver(Client client, Supplier<ClusterState> clusterState, IndexNameExpressionResolver expressionResolver) {
        this.client = client;
        this.clusterState = clusterState;
        this.expressionResolver = expressionResolver;
    }

    public void resolve(
        String expression,
        int maxIndices,
        TaskId parentTask,
        BooleanSupplier cancelled,
        BooleanSupplier expired,
        ActionListener<List<ResolvedIndex>> listener
    ) {
        List<String> tokens = tokens(expression);
        getMappings(tokens, parentTask, ActionListener.wrap(mappings -> {
            ClusterState state = clusterState.get();
            Targets targets = targets(state, tokens);
            Map<String, MappingMetadata> visible = new TreeMap<>();
            mappings.forEach((name, mapping) -> {
                if (targets.kept.contains(name) || !isInternal(state, name)) {
                    visible.put(name, mapping);
                }
            });
            if (mappings.size() == 1 && visible.size() == 1) {
                // A target that resolves to one index is searched as given, exactly as for a single index.
                Map.Entry<String, MappingMetadata> only = visible.entrySet().iterator().next();
                MappingMetadata mapping = only.getValue();
                listener
                    .onResponse(List.of(new ResolvedIndex(only.getKey(), expression, false, properties(mapping), description(mapping))));
                return;
            }
            finish(expression, tokens, visible, targets, maxIndices, listener);
        }, e -> {
            if (isForbidden(e) && tokens.stream().anyMatch(IndexResolver::isPattern)) {
                resolveReadable(expression, tokens, maxIndices, parentTask, cancelled, expired, listener);
            } else {
                listener.onFailure(e);
            }
        }));
    }

    private void resolveReadable(
        String expression,
        List<String> tokens,
        int maxIndices,
        TaskId parentTask,
        BooleanSupplier cancelled,
        BooleanSupplier expired,
        ActionListener<List<ResolvedIndex>> listener
    ) {
        List<String> named = tokens.stream().filter(t -> !isPattern(t) && !t.startsWith("-")).toList();
        ActionListener<Map<String, MappingMetadata>> afterNamed = ActionListener.wrap(namedMappings -> {
            ClusterState state = clusterState.get();
            Targets targets = targets(state, tokens);
            String[] patterns = tokens.stream().filter(t -> !named.contains(t)).toArray(String[]::new);
            Set<String> candidates = new TreeSet<>();
            for (String name : expressionResolver.concreteIndexNames(state, IndicesOptions.lenientExpandOpen(), true, patterns)) {
                if (!namedMappings.containsKey(name) && (targets.kept.contains(name) || !isInternal(state, name))) {
                    candidates.add(name);
                }
            }
            // Indices a pattern matches by name are checked on their own; the others through the matched alias or data
            // stream, since the caller may be allowed to read them only through it. Each check credits only the
            // candidates assigned to it, so an index is kept only when its own check succeeds and is counted once.
            Map<String, Set<String>> checks = new TreeMap<>();
            for (String candidate : candidates) {
                String via = targets.viaPattern.get(candidate);
                String check = via == null || matchesPattern(targets.patterns, candidate) ? candidate : via;
                checks.computeIfAbsent(check, k -> new HashSet<>()).add(candidate);
            }
            if (checks.size() > CHECK_LIMIT_FACTOR * maxIndices) {
                listener.onFailure(tooManyToCheck(expression, maxIndices));
                return;
            }
            AtomicInteger readable = new AtomicInteger(namedMappings.size());
            AtomicBoolean skippedAtDeadline = new AtomicBoolean();
            BoundedFanOut.<String, Map<String, MappingMetadata>>run(List.copyOf(checks.keySet()), CHECK_CONCURRENCY, (check, checked) -> {
                if (cancelled.getAsBoolean() || readable.get() > maxIndices) {
                    checked.onResponse(Map.of());
                    return;
                }
                if (expired.getAsBoolean()) {
                    skippedAtDeadline.set(true);
                    checked.onResponse(Map.of());
                    return;
                }
                getMappings(List.of(check), parentTask, ActionListener.wrap(m -> {
                    Map<String, MappingMetadata> kept = new TreeMap<>(m);
                    kept.keySet().retainAll(checks.get(check));
                    readable.addAndGet(kept.size());
                    checked.onResponse(kept);
                }, checked::onFailure));
            }, outcomes -> {
                if (cancelled.getAsBoolean()) {
                    listener.onFailure(new TaskCancelledException("The agentic index request was cancelled"));
                    return;
                }
                if (skippedAtDeadline.get() && readable.get() <= maxIndices) {
                    listener.onFailure(timedOut(expression));
                    return;
                }
                Map<String, MappingMetadata> visible = new TreeMap<>(namedMappings);
                for (Outcome<Map<String, MappingMetadata>> outcome : outcomes) {
                    if (outcome.failure() == null) {
                        visible.putAll(outcome.result());
                    } else if (!isForbidden(outcome.failure()) && !isNotFound(outcome.failure())) {
                        listener.onFailure(outcome.failure());
                        return;
                    }
                }
                finish(expression, tokens, visible, targets, maxIndices, listener);
            });
        }, listener::onFailure);
        if (named.isEmpty()) {
            afterNamed.onResponse(Map.of());
        } else {
            getMappings(named, parentTask, afterNamed);
        }
    }

    private void finish(
        String expression,
        List<String> tokens,
        Map<String, MappingMetadata> visible,
        Targets targets,
        int maxIndices,
        ActionListener<List<ResolvedIndex>> listener
    ) {
        if (visible.isEmpty()) {
            listener
                .onFailure(new OpenSearchStatusException("No indices that you can read match [" + expression + "]", RestStatus.NOT_FOUND));
            return;
        }
        if (visible.size() > maxIndices) {
            listener.onFailure(tooManyIndices(expression, maxIndices));
            return;
        }
        List<ResolvedIndex> resolved = new ArrayList<>();
        visible.forEach((name, mapping) -> {
            String via = targets.viaExplicit.getOrDefault(name, targets.viaPattern.get(name));
            boolean direct = targets.direct.contains(name) || matchesPattern(targets.patterns, name);
            String target = direct || via == null ? name : via;
            resolved.add(new ResolvedIndex(name, target, !target.equals(name), properties(mapping), description(mapping)));
        });
        listener.onResponse(resolved);
    }

    static OpenSearchStatusException tooManyIndices(String expression, int maxIndices) {
        return new OpenSearchStatusException(
            String
                .format(
                    Locale.ROOT,
                    "[%s] resolves to more than %d indices that you can read, the limit set by %s. Use a narrower target.",
                    expression,
                    maxIndices,
                    ML_COMMONS_AGENTIC_INDEX_MAX_INDICES.getKey()
                ),
            RestStatus.BAD_REQUEST
        );
    }

    private static OpenSearchStatusException tooManyToCheck(String expression, int maxIndices) {
        return new OpenSearchStatusException(
            String
                .format(
                    Locale.ROOT,
                    "[%s] matches indices that you cannot read, and more than %d indices to check one by one (%d times %s). "
                        + "Use a narrower target that matches only indices you can read.",
                    expression,
                    CHECK_LIMIT_FACTOR * maxIndices,
                    CHECK_LIMIT_FACTOR,
                    ML_COMMONS_AGENTIC_INDEX_MAX_INDICES.getKey()
                ),
            RestStatus.BAD_REQUEST
        );
    }

    private static OpenSearchStatusException timedOut(String expression) {
        return new TimedOutException(
            "The request timed out while checking which indices that ["
                + expression
                + "] matches you can read. "
                + "Use a narrower target or a longer timeout."
        );
    }

    /** The deadline passed before every index the target matches was checked for read access. */
    public static final class TimedOutException extends OpenSearchStatusException {
        TimedOutException(String message) {
            super(message, RestStatus.REQUEST_TIMEOUT);
        }
    }

    private void getMappings(List<String> names, TaskId parentTask, ActionListener<Map<String, MappingMetadata>> listener) {
        GetMappingsRequest request = new GetMappingsRequest().indices(names.toArray(new String[0]));
        request.setParentTask(parentTask);
        client.admin().indices().getMappings(request, ActionListener.wrap(response -> {
            Map<String, MappingMetadata> mappings = response.getMappings();
            listener.onResponse(mappings == null ? Map.of() : mappings);
        }, listener::onFailure));
    }

    /**
     * How the target's tokens reach indices. {@code kept} indices stay even when hidden or dot-prefixed: those named
     * explicitly or through an explicit alias or data stream, and backing indices of a data stream a pattern matches.
     * {@code direct} indices are named explicitly. {@code viaExplicit} and {@code viaPattern} map an index to the
     * alias or data stream it is reached through, named explicitly or matched by a pattern.
     */
    private record Targets(Set<String> kept, Set<String> direct, Map<String, String> viaExplicit, Map<String, String> viaPattern,
        List<String> patterns) {
    }

    private static Targets targets(ClusterState state, List<String> tokens) {
        Set<String> kept = new HashSet<>();
        Set<String> direct = new HashSet<>();
        Map<String, String> viaExplicit = new HashMap<>();
        Map<String, String> viaPattern = new HashMap<>();
        List<String> patterns = tokens.stream().filter(t -> isPattern(t) && !t.startsWith("-")).toList();
        SortedMap<String, IndexAbstraction> lookup = state.metadata().getIndicesLookup();
        for (String token : tokens) {
            if (isPattern(token) || token.startsWith("-")) {
                continue;
            }
            IndexAbstraction abstraction = lookup.get(token);
            if (abstraction == null || abstraction.getType() == IndexAbstraction.Type.CONCRETE_INDEX) {
                kept.add(token);
                direct.add(token);
                continue;
            }
            for (IndexMetadata metadata : abstraction.getIndices()) {
                String name = metadata.getIndex().getName();
                kept.add(name);
                viaExplicit.putIfAbsent(name, token);
            }
        }
        if (!patterns.isEmpty()) {
            // Sorted lookup, so an index in several matched aliases is always reached through the same one.
            for (IndexAbstraction abstraction : lookup.values()) {
                IndexAbstraction.Type type = abstraction.getType();
                String name = abstraction.getName();
                if (type == IndexAbstraction.Type.CONCRETE_INDEX
                    || name.startsWith(".")
                    || abstraction.isHidden()
                    || !matchesPattern(patterns, name)) {
                    continue;
                }
                for (IndexMetadata metadata : abstraction.getIndices()) {
                    String index = metadata.getIndex().getName();
                    viaPattern.putIfAbsent(index, name);
                    if (type == IndexAbstraction.Type.DATA_STREAM) {
                        kept.add(index);
                    }
                }
            }
        }
        return new Targets(kept, direct, viaExplicit, viaPattern, patterns);
    }

    private static boolean matchesPattern(List<String> patterns, String name) {
        return patterns.stream().anyMatch(p -> "_all".equals(p) || Regex.simpleMatch(p, name));
    }

    static List<String> tokens(String expression) {
        return Arrays.stream(expression.split(",")).map(String::trim).filter(t -> !t.isEmpty()).toList();
    }

    static boolean isPattern(String token) {
        return Regex.isSimpleMatchPattern(token) || "_all".equals(token);
    }

    private static boolean isInternal(ClusterState state, String name) {
        if (name.startsWith(".")) {
            return true;
        }
        IndexMetadata metadata = state.metadata().index(name);
        return metadata != null && (metadata.isSystem() || IndexMetadata.INDEX_HIDDEN_SETTING.get(metadata.getSettings()));
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> properties(MappingMetadata mapping) {
        return mapping == null ? null : (Map<String, Object>) mapping.getSourceAsMap().get("properties");
    }

    private static String description(MappingMetadata mapping) {
        Object meta = mapping == null ? null : mapping.getSourceAsMap().get("_meta");
        Object description = meta instanceof Map<?, ?> m ? m.get("description") : null;
        return description instanceof String text ? text : null;
    }

    private static boolean isForbidden(Exception e) {
        return ExceptionsHelper.status(ExceptionsHelper.unwrapCause(e)) == RestStatus.FORBIDDEN;
    }

    private static boolean isNotFound(Exception e) {
        return ExceptionsHelper.status(ExceptionsHelper.unwrapCause(e)) == RestStatus.NOT_FOUND;
    }
}
