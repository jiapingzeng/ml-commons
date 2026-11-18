/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import java.util.ArrayList;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import org.opensearch.search.aggregations.AggregationBuilders;
import org.opensearch.search.aggregations.Aggregations;
import org.opensearch.search.aggregations.bucket.terms.Terms;
import org.opensearch.search.aggregations.metrics.Max;
import org.opensearch.search.aggregations.metrics.Min;
import org.opensearch.search.builder.SearchSourceBuilder;

import lombok.Getter;

/**
 * Compact, LLM-facing description of an index: which fields can be filtered exactly, which can be
 * matched as text, and the observed values of low-cardinality keyword fields and date ranges.
 * Only the first {@link #MAX_PROFILED_FIELDS} fields in preference order are profiled, so cost and prompt size stay bounded
 * for wide mappings.
 */
@Getter
public class IndexProfile {

    public enum FieldKind {
        KEYWORD,
        TEXT,
        DATE,
        NUMBER,
        BOOLEAN,
        VECTOR
    }

    static final int MAX_TERMS = 25;
    static final int MAX_PROFILED_FIELDS = 50;
    static final int MAX_VALUE_CHARS = 100;
    static final int MAX_RENDERED_CHARS = 8000;
    // Profiling preference: searchable text first, then likely low-cardinality filters; keyword sub-fields of text
    // (usually titles and other high-cardinality values) come last.
    private static final List<FieldKind> PROFILE_ORDER = List
        .of(FieldKind.TEXT, FieldKind.KEYWORD, FieldKind.DATE, FieldKind.BOOLEAN, FieldKind.NUMBER);
    private static final String TERMS_AGG_PREFIX = "terms__";
    private static final String MIN_AGG_PREFIX = "min__";
    private static final String MAX_AGG_PREFIX = "max__";
    private static final Set<String> NUMBER_TYPES = Set
        .of("long", "integer", "short", "byte", "double", "float", "half_float", "scaled_float", "unsigned_long");

    private final String index;
    // Field path to kind; text fields with a keyword sub-field also get "<path>.keyword" as KEYWORD.
    private final Map<String, FieldKind> fields;
    // Fields with doc_values disabled; aggregations and sorts on them fail.
    private final Set<String> noDocValues;
    // Fields described to the LLM, at most MAX_PROFILED_FIELDS.
    private final List<String> profiledFields;
    private final Map<String, List<String>> keywordValues;
    private final Map<String, String[]> dateRanges;
    private final String indexInsight;

    private IndexProfile(
        String index,
        Map<String, FieldKind> fields,
        Set<String> noDocValues,
        List<String> profiledFields,
        Map<String, List<String>> keywordValues,
        Map<String, String[]> dateRanges,
        String indexInsight
    ) {
        this.index = index;
        this.fields = fields;
        this.noDocValues = noDocValues;
        this.profiledFields = profiledFields;
        this.keywordValues = keywordValues;
        this.dateRanges = dateRanges;
        this.indexInsight = indexInsight;
    }

    /**
     * Builds the field inventory from the "properties" section of an index mapping.
     */
    public static IndexProfile fromMappingProperties(String index, Map<String, Object> properties) {
        return fromMappingProperties(index, properties, List.of());
    }

    /**
     * Builds the field inventory, profiling the given text fields first.
     */
    public static IndexProfile fromMappingProperties(String index, Map<String, Object> properties, List<String> preferredFields) {
        Map<String, FieldKind> fields = new LinkedHashMap<>();
        Set<String> noDocValues = new HashSet<>();
        Set<String> subFields = new LinkedHashSet<>();
        collectFields("", properties, fields, noDocValues, subFields);
        return new IndexProfile(
            index,
            fields,
            noDocValues,
            selectProfiledFields(fields, subFields, preferredFields),
            new LinkedHashMap<>(),
            new LinkedHashMap<>(),
            null
        );
    }

    private static List<String> selectProfiledFields(Map<String, FieldKind> fields, Set<String> subFields, List<String> preferred) {
        Set<String> selected = new LinkedHashSet<>();
        for (String field : preferred) {
            if (fields.containsKey(field)) {
                selected.add(field);
            }
        }
        for (FieldKind kind : PROFILE_ORDER) {
            fields.forEach((name, k) -> {
                if (k == kind && !subFields.contains(name)) {
                    selected.add(name);
                }
            });
        }
        selected.addAll(subFields);
        return List.copyOf(selected).subList(0, Math.min(MAX_PROFILED_FIELDS, selected.size()));
    }

    /**
     * Copy of this profile with index insight text added to the rendered description.
     */
    public IndexProfile withIndexInsight(String insight) {
        return new IndexProfile(index, fields, noDocValues, profiledFields, keywordValues, dateRanges, insight);
    }

    @SuppressWarnings("unchecked")
    private static void collectFields(
        String prefix,
        Map<String, Object> properties,
        Map<String, FieldKind> out,
        Set<String> noDocValues,
        Set<String> subFieldsOut
    ) {
        if (properties == null) {
            return;
        }
        for (Map.Entry<String, Object> entry : properties.entrySet()) {
            String path = prefix.isEmpty() ? entry.getKey() : prefix + "." + entry.getKey();
            Map<String, Object> def = (Map<String, Object>) entry.getValue();
            String type = (String) def.get("type");
            if (def.containsKey("properties") && (type == null || "object".equals(type))) {
                collectFields(path, (Map<String, Object>) def.get("properties"), out, noDocValues, subFieldsOut);
                continue;
            }
            FieldKind kind = kindOfType(type);
            if (kind == null) {
                continue;
            }
            out.put(path, kind);
            if (docValuesDisabled(def)) {
                noDocValues.add(path);
            }
            Map<String, Object> subFields = (Map<String, Object>) def.get("fields");
            if (subFields != null) {
                for (Map.Entry<String, Object> sub : subFields.entrySet()) {
                    Map<String, Object> subDef = (Map<String, Object>) sub.getValue();
                    if (kindOfType((String) subDef.get("type")) == FieldKind.KEYWORD) {
                        String subPath = path + "." + sub.getKey();
                        out.put(subPath, FieldKind.KEYWORD);
                        subFieldsOut.add(subPath);
                        if (docValuesDisabled(subDef)) {
                            noDocValues.add(subPath);
                        }
                    }
                }
            }
        }
    }

    private static boolean docValuesDisabled(Map<String, Object> def) {
        return "false".equals(String.valueOf(def.get("doc_values")));
    }

    private static FieldKind kindOfType(String type) {
        if (type == null) {
            return null;
        }
        if ("keyword".equals(type) || "constant_keyword".equals(type)) {
            return FieldKind.KEYWORD;
        }
        if ("text".equals(type) || "match_only_text".equals(type)) {
            return FieldKind.TEXT;
        }
        if ("date".equals(type) || "date_nanos".equals(type)) {
            return FieldKind.DATE;
        }
        if ("boolean".equals(type)) {
            return FieldKind.BOOLEAN;
        }
        if ("knn_vector".equals(type)) {
            return FieldKind.VECTOR;
        }
        return NUMBER_TYPES.contains(type) ? FieldKind.NUMBER : null;
    }

    public List<String> fieldsOfKind(FieldKind kind) {
        List<String> result = new ArrayList<>();
        fields.forEach((name, k) -> {
            if (k == kind) {
                result.add(name);
            }
        });
        return result;
    }

    /**
     * Profiled fields of the given kind, in profile order.
     */
    public List<String> profiledFieldsOfKind(FieldKind kind) {
        return profiledFields.stream().filter(f -> fields.get(f) == kind).toList();
    }

    public FieldKind kindOf(String field) {
        return fields.get(field);
    }

    public boolean hasDocValues(String field) {
        return !noDocValues.contains(field);
    }

    /**
     * Aggregations that collect keyword values and date ranges of the profiled fields that have doc values.
     */
    public SearchSourceBuilder statsQuery() {
        SearchSourceBuilder source = new SearchSourceBuilder().size(0).trackTotalHits(false);
        for (String field : aggregatable(FieldKind.KEYWORD)) {
            source.aggregation(AggregationBuilders.terms(TERMS_AGG_PREFIX + field).field(field).size(MAX_TERMS));
        }
        for (String field : aggregatable(FieldKind.DATE)) {
            source.aggregation(AggregationBuilders.min(MIN_AGG_PREFIX + field).field(field));
            source.aggregation(AggregationBuilders.max(MAX_AGG_PREFIX + field).field(field));
        }
        return source;
    }

    private List<String> aggregatable(FieldKind kind) {
        return profiledFieldsOfKind(kind).stream().filter(this::hasDocValues).toList();
    }

    public void applyStats(Aggregations aggregations) {
        if (aggregations == null) {
            return;
        }
        for (String field : aggregatable(FieldKind.KEYWORD)) {
            Terms terms = aggregations.get(TERMS_AGG_PREFIX + field);
            if (terms != null && !terms.getBuckets().isEmpty()) {
                List<String> values = new ArrayList<>();
                terms.getBuckets().forEach(b -> values.add(PassageStore.truncate(b.getKeyAsString(), MAX_VALUE_CHARS)));
                keywordValues.put(field, values);
            }
        }
        for (String field : aggregatable(FieldKind.DATE)) {
            Min min = aggregations.get(MIN_AGG_PREFIX + field);
            Max max = aggregations.get(MAX_AGG_PREFIX + field);
            if (min != null && max != null && Double.isFinite(min.getValue()) && Double.isFinite(max.getValue())) {
                dateRanges.put(field, new String[] { min.getValueAsString(), max.getValueAsString() });
            }
        }
    }

    /**
     * Renders the profile as plain text for the planner prompt, capped at {@link #MAX_RENDERED_CHARS}.
     */
    public String render() {
        StringBuilder sb = new StringBuilder();
        sb.append("Index: ").append(index).append('\n');
        appendList(sb, "Text fields (keyword/semantic match)", profiledFieldsOfKind(FieldKind.TEXT));
        sb.append("Exact-match keyword fields:\n");
        for (String field : profiledFieldsOfKind(FieldKind.KEYWORD)) {
            List<String> values = keywordValues.get(field);
            sb.append("- ").append(field);
            if (values != null) {
                boolean truncated = values.size() >= MAX_TERMS;
                sb.append(" values: ").append(String.join(" | ", values)).append(truncated ? " | ..." : "");
            }
            sb.append('\n');
        }
        sb.append("Date fields:\n");
        for (String field : profiledFieldsOfKind(FieldKind.DATE)) {
            String[] range = dateRanges.get(field);
            sb.append("- ").append(field);
            if (range != null) {
                sb.append(String.format(Locale.ROOT, " (from %s to %s)", range[0], range[1]));
            }
            sb.append('\n');
        }
        appendList(sb, "Numeric fields", profiledFieldsOfKind(FieldKind.NUMBER));
        appendList(sb, "Boolean fields", profiledFieldsOfKind(FieldKind.BOOLEAN));
        long omitted = fields.values().stream().filter(k -> k != FieldKind.VECTOR).count() - profiledFields.size();
        if (omitted > 0) {
            sb.append(String.format(Locale.ROOT, "(%d more fields not profiled)\n", omitted));
        }
        if (indexInsight != null && !indexInsight.isBlank()) {
            sb.append("Index insight:\n").append(indexInsight).append('\n');
        }
        return sb.length() <= MAX_RENDERED_CHARS ? sb.toString() : sb.substring(0, MAX_RENDERED_CHARS) + "\n... (profile truncated)\n";
    }

    private static void appendList(StringBuilder sb, String label, List<String> values) {
        if (!values.isEmpty()) {
            sb.append(label).append(": ").append(String.join(", ", values)).append('\n');
        }
    }
}
