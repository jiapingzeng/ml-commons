/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import java.io.IOException;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.MultiMatchQueryBuilder;
import org.opensearch.index.query.QueryBuilder;
import org.opensearch.index.query.QueryBuilders;
import org.opensearch.index.query.RangeQueryBuilder;
import org.opensearch.ml.action.agenticindex.IndexProfile.FieldKind;

import lombok.Getter;

/**
 * One planned retrieval step: optional text to match plus exact filters, validated against the index profile.
 * Planner output is model-generated, so invalid parts are dropped and reported as warnings instead of failing the request.
 */
@Getter
public class SubQuery {

    public enum SearchMode {
        KEYWORD,
        SEMANTIC,
        HYBRID
    }

    private static final Set<String> RANGE_OPS = Set.of("gte", "gt", "lte", "lt");

    // Concrete index the subquery searches.
    private final String index;
    private final String text;
    private final SearchMode mode;
    // Field to accepted filter value: a scalar, a list of scalars, or a range map.
    private final Map<String, Object> filters;
    private final List<String> warnings;

    SubQuery(String text, SearchMode mode, Map<String, Object> filters, List<String> warnings) {
        this(null, text, mode, filters, warnings);
    }

    SubQuery(String index, String text, SearchMode mode, Map<String, Object> filters, List<String> warnings) {
        this.index = index;
        this.text = text;
        this.mode = mode;
        this.filters = filters;
        this.warnings = warnings;
    }

    /**
     * Parses one planner item such as {"query": "...", "search": "keyword", "filters": {"plan": ["Pro"]}}.
     * Returns null when the item has neither usable text nor usable filters.
     */
    public static SubQuery fromPlan(Object item, IndexProfile profile, boolean semanticAvailable) {
        return fromPlan(item, null, profile, semanticAvailable, "semantic search not configured, used keyword");
    }

    /**
     * Parses one planner item for the given concrete index; filters are validated against that index's profile.
     * @param semanticUnavailableNote warning added when the item asks for semantic or hybrid search that is unavailable
     */
    @SuppressWarnings("unchecked")
    public static SubQuery fromPlan(
        Object item,
        String index,
        IndexProfile profile,
        boolean semanticAvailable,
        String semanticUnavailableNote
    ) {
        if (!(item instanceof Map)) {
            return null;
        }
        Map<String, Object> map = (Map<String, Object>) item;
        List<String> warnings = new ArrayList<>();
        Object rawText = map.get("query");
        String text = rawText == null ? "" : String.valueOf(rawText).trim();
        SearchMode mode = parseMode(map.get("search"), semanticAvailable, semanticUnavailableNote, warnings);

        Map<String, Object> filters = new LinkedHashMap<>();
        Object rawFilters = map.get("filters");
        if (rawFilters instanceof Map) {
            for (Map.Entry<String, Object> entry : ((Map<String, Object>) rawFilters).entrySet()) {
                addFilter(entry.getKey(), entry.getValue(), profile, filters, warnings);
            }
        } else if (rawFilters != null) {
            warnings.add("ignored filters that are not an object");
        }
        if (text.isEmpty() && filters.isEmpty()) {
            return null;
        }
        return new SubQuery(index, text, mode, filters, warnings);
    }

    private static SearchMode parseMode(Object raw, boolean semanticAvailable, String unavailableNote, List<String> warnings) {
        SearchMode mode = SearchMode.KEYWORD;
        if (raw != null) {
            try {
                mode = SearchMode.valueOf(String.valueOf(raw).trim().toUpperCase(Locale.ROOT));
            } catch (IllegalArgumentException e) {
                warnings.add("unknown search mode " + raw + ", used keyword");
            }
        } else if (semanticAvailable) {
            mode = SearchMode.HYBRID;
        }
        if (mode != SearchMode.KEYWORD && !semanticAvailable) {
            warnings.add(unavailableNote);
            mode = SearchMode.KEYWORD;
        }
        return mode;
    }

    @SuppressWarnings("unchecked")
    private static void addFilter(String field, Object value, IndexProfile profile, Map<String, Object> out, List<String> warnings) {
        FieldKind kind = profile.kindOf(field);
        if (kind == FieldKind.TEXT && profile.kindOf(field + ".keyword") == FieldKind.KEYWORD) {
            field = field + ".keyword";
            kind = FieldKind.KEYWORD;
        }
        if (kind == null || kind == FieldKind.TEXT || kind == FieldKind.VECTOR) {
            warnings.add("dropped filter on non-filterable field " + field);
            return;
        }
        if (value == null) {
            warnings.add("dropped empty filter on " + field);
            return;
        }
        if (value instanceof Map) {
            Map<String, Object> range = new LinkedHashMap<>();
            for (Map.Entry<String, Object> op : ((Map<String, Object>) value).entrySet()) {
                if (RANGE_OPS.contains(op.getKey()) && op.getValue() != null) {
                    range.put(op.getKey(), normalize(op.getValue()));
                }
            }
            if (range.isEmpty() || kind == FieldKind.KEYWORD || kind == FieldKind.BOOLEAN) {
                warnings.add("dropped invalid range filter on " + field);
                return;
            }
            out.put(field, range);
        } else if (value instanceof List) {
            List<Object> values = new ArrayList<>();
            for (Object v : (List<Object>) value) {
                if (v != null) {
                    values.add(normalize(v));
                }
            }
            if (values.isEmpty()) {
                warnings.add("dropped empty filter on " + field);
                return;
            }
            out.put(field, values.size() == 1 ? values.get(0) : values);
        } else {
            out.put(field, normalize(value));
        }
    }

    // Gson parses all JSON numbers as doubles; keep integral values integral so term filters match.
    private static Object normalize(Object value) {
        if (value instanceof Double d && d == Math.rint(d) && !d.isInfinite()) {
            return d.longValue();
        }
        return value;
    }

    public BoolQueryBuilder filterQuery() {
        BoolQueryBuilder bool = QueryBuilders.boolQuery();
        for (Map.Entry<String, Object> entry : filters.entrySet()) {
            bool.filter(filterClause(entry.getKey(), entry.getValue()));
        }
        return bool;
    }

    @SuppressWarnings("unchecked")
    private static QueryBuilder filterClause(String field, Object value) {
        if (value instanceof Map) {
            RangeQueryBuilder range = QueryBuilders.rangeQuery(field);
            ((Map<String, Object>) value).forEach((op, v) -> {
                switch (op) {
                    case "gte" -> range.gte(v);
                    case "gt" -> range.gt(v);
                    case "lte" -> range.lte(v);
                    case "lt" -> range.lt(v);
                    default -> throw new IllegalStateException("unexpected range op " + op);
                }
            });
            return range;
        }
        if (value instanceof List) {
            return QueryBuilders.termsQuery(field, (List<Object>) value);
        }
        return QueryBuilders.termQuery(field, value);
    }

    /**
     * BM25 query over the text fields, with filters. With no text, returns a filter-only query.
     */
    public QueryBuilder keywordQuery(List<String> textFields) {
        BoolQueryBuilder bool = filterQuery();
        if (!text.isEmpty()) {
            bool.must(QueryBuilders.multiMatchQuery(text, textFields.toArray(new String[0])).type(MultiMatchQueryBuilder.Type.BEST_FIELDS));
        }
        return bool;
    }

    /**
     * Neural query with the filters applied inside the k-NN search, wrapped because the neural query builder
     * lives in the neural-search plugin.
     */
    public QueryBuilder semanticQuery(String embeddingField, String embeddingModelId, int k) throws IOException {
        try (XContentBuilder builder = JsonXContent.contentBuilder()) {
            builder.startObject().startObject("neural").startObject(embeddingField);
            builder.field("query_text", text).field("model_id", embeddingModelId).field("k", k);
            if (!filters.isEmpty()) {
                builder.field("filter");
                filterQuery().toXContent(builder, ToXContent.EMPTY_PARAMS);
            }
            builder.endObject().endObject().endObject();
            return QueryBuilders.wrapperQuery(BytesReference.bytes(builder).utf8ToString());
        }
    }

    public Map<String, Object> toMap() {
        Map<String, Object> map = new LinkedHashMap<>();
        if (index != null) {
            map.put("index", index);
        }
        map.put("query", text);
        map.put("search", mode.name().toLowerCase(Locale.ROOT));
        map.put("filters", filters);
        if (!warnings.isEmpty()) {
            map.put("warnings", warnings);
        }
        return map;
    }
}
