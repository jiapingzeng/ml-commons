/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import static org.opensearch.ml.common.utils.StringUtils.gson;

import java.util.ArrayList;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.function.Function;

import org.opensearch.common.xcontent.support.XContentMapValues;
import org.opensearch.search.SearchHit;

/**
 * Deduplicated passages retrieved across all subqueries and rounds, numbered P1, P2, ... in first-seen order
 * so the LLM can cite them by id.
 */
public class PassageStore {

    static final int MAX_TEXT_CHARS = 1500;
    static final int MAX_METADATA_CHARS = 200;
    static final int MAX_METADATA_ENTRIES = 20;
    static final int MAX_METADATA_TOTAL_CHARS = 1000;
    // Metadata fields surfaced as citation location when present in the source document.
    static final List<String> LOCATION_FIELDS = List.of("source", "file", "file_name", "url", "page", "page_number", "chunk", "chunk_id");

    // Text and excluded fields of a passage's index, by concrete index name.
    private final Function<String, List<String>> textFieldsOf;
    private final Function<String, Set<String>> excludedFieldsOf;
    private final int maxPassages;
    // Whether rendered passages name their index, for requests over several indices.
    private final boolean showIndex;
    private final Map<String, Map<String, Object>> byKey = new LinkedHashMap<>();
    private final Map<String, Map<String, Object>> byId = new LinkedHashMap<>();

    public PassageStore(List<String> textFields, Set<String> excludedFields, int maxPassages) {
        this(index -> textFields, index -> excludedFields, maxPassages, false);
    }

    public PassageStore(
        Function<String, List<String>> textFieldsOf,
        Function<String, Set<String>> excludedFieldsOf,
        int maxPassages,
        boolean showIndex
    ) {
        this.textFieldsOf = textFieldsOf;
        this.excludedFieldsOf = excludedFieldsOf;
        this.maxPassages = maxPassages;
        this.showIndex = showIndex;
    }

    /**
     * Adds a hit if new and returns its passage id, or null when the store is full.
     */
    public String add(SearchHit hit, int round, int subQueryIndex) {
        String key = hit.getIndex() + "/" + hit.getId();
        Map<String, Object> existing = byKey.get(key);
        if (existing != null) {
            return (String) existing.get("passage_id");
        }
        if (byKey.size() >= maxPassages) {
            return null;
        }
        String passageId = "P" + (byKey.size() + 1);
        Map<String, Object> source = hit.getSourceAsMap() == null ? Map.of() : hit.getSourceAsMap();
        Map<String, Object> passage = new LinkedHashMap<>();
        passage.put("passage_id", passageId);
        passage.put("index", hit.getIndex());
        passage.put("doc_id", hit.getId());
        passage.put("score", Float.isNaN(hit.getScore()) ? null : hit.getScore());
        passage.put("round", round);
        passage.put("subquery", subQueryIndex);
        if (source.get("title") != null) {
            passage.put("title", truncate(String.valueOf(source.get("title")), MAX_METADATA_CHARS));
        }
        List<String> textFields = textFieldsOf.apply(hit.getIndex());
        passage.put("text", truncate(passageText(source, textFields), MAX_TEXT_CHARS));
        passage.put("metadata", metadata(source, textFields, excludedFieldsOf.apply(hit.getIndex())));
        byKey.put(key, passage);
        byId.put(passageId, passage);
        return passageId;
    }

    // Location fields first, then the rest by name, until the entry or character budget is used up.
    private static Map<String, Object> metadata(Map<String, Object> source, List<String> textFields, Set<String> excludedFields) {
        List<String> keys = new ArrayList<>();
        LOCATION_FIELDS.stream().filter(source::containsKey).forEach(keys::add);
        source.keySet().stream().filter(k -> !LOCATION_FIELDS.contains(k)).sorted().forEach(keys::add);
        Map<String, Object> metadata = new LinkedHashMap<>();
        int chars = 0;
        for (String key : keys) {
            Object raw = source.get(key);
            if (raw == null || coveredByTextOrExcluded(key, textFields, excludedFields)) {
                continue;
            }
            if (metadata.size() >= MAX_METADATA_ENTRIES) {
                break;
            }
            Object value = metadataValue(raw);
            int size = key.length() + (value instanceof String s ? s.length() : gson.toJson(value).length());
            if (chars + size <= MAX_METADATA_TOTAL_CHARS) {
                metadata.put(key, value);
                chars += size;
            }
        }
        return metadata;
    }

    // A top-level key is left out of metadata when it is, or contains, a text or excluded field (content for content.body).
    private static boolean coveredByTextOrExcluded(String key, List<String> textFields, Set<String> excludedFields) {
        for (Collection<String> fields : List.of(textFields, excludedFields)) {
            for (String field : fields) {
                if (field.equals(key) || field.startsWith(key + ".")) {
                    return true;
                }
            }
        }
        return false;
    }

    // Numbers and booleans pass through; strings and objects are capped so one large field cannot crowd out the prompt.
    private static Object metadataValue(Object value) {
        if (value instanceof Number || value instanceof Boolean) {
            return value;
        }
        if (value instanceof String s) {
            return truncate(s, MAX_METADATA_CHARS);
        }
        String json = gson.toJson(value);
        return json.length() <= MAX_METADATA_CHARS ? value : truncate(json, MAX_METADATA_CHARS);
    }

    private static String passageText(Map<String, Object> source, List<String> textFields) {
        StringBuilder sb = new StringBuilder();
        for (String field : textFields) {
            Object value = XContentMapValues.extractValue(field, source);
            if (value instanceof List<?> values) {
                values.stream().filter(v -> v != null).forEach(v -> appendLine(sb, v));
            } else if (value != null) {
                appendLine(sb, value);
            }
        }
        return sb.toString();
    }

    private static void appendLine(StringBuilder sb, Object value) {
        if (sb.length() > 0) {
            sb.append('\n');
        }
        sb.append(value);
    }

    static String truncate(String value, int max) {
        return value.length() <= max ? value : value.substring(0, max) + "...";
    }

    public Map<String, Object> get(String passageId) {
        return byId.get(passageId);
    }

    public Collection<Map<String, Object>> all() {
        return byId.values();
    }

    public int size() {
        return byId.size();
    }

    /**
     * Citation view of a passage: ids plus the location fields a reader needs to open the source.
     */
    @SuppressWarnings("unchecked")
    public Map<String, Object> citation(String passageId) {
        Map<String, Object> passage = byId.get(passageId);
        Map<String, Object> citation = new LinkedHashMap<>();
        citation.put("passage_id", passageId);
        citation.put("index", passage.get("index"));
        citation.put("doc_id", passage.get("doc_id"));
        Map<String, Object> metadata = (Map<String, Object>) passage.get("metadata");
        for (String field : LOCATION_FIELDS) {
            if (metadata.containsKey(field)) {
                citation.put(field, metadata.get(field));
            }
        }
        if (passage.containsKey("title")) {
            citation.put("title", passage.get("title"));
        }
        return citation;
    }

    /**
     * Renders passages for an LLM prompt, one block per passage.
     */
    public String render(Collection<Map<String, Object>> passages) {
        List<String> labels = new ArrayList<>();
        passages.forEach(p -> labels.add((String) p.get("passage_id")));
        return render(passages, labels);
    }

    /**
     * Renders passages under the given labels. The answer step labels its subset P1..Pn by position because
     * the model tends to cite by position when ids have gaps.
     */
    @SuppressWarnings("unchecked")
    public String render(Collection<Map<String, Object>> passages, List<String> labels) {
        List<String> blocks = new ArrayList<>();
        int i = 0;
        for (Map<String, Object> passage : passages) {
            StringBuilder sb = new StringBuilder();
            sb.append('[').append(labels.get(i++)).append(']');
            if (showIndex) {
                sb.append(" index=").append(passage.get("index"));
            }
            sb.append(" doc_id=").append(passage.get("doc_id"));
            Map<String, Object> metadata = (Map<String, Object>) passage.get("metadata");
            metadata.forEach((k, v) -> sb.append(' ').append(k).append('=').append(v));
            sb.append('\n').append(passage.get("text"));
            blocks.add(sb.toString());
        }
        return String.join("\n\n", blocks);
    }
}
