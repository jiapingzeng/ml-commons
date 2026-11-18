/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import static org.mockito.Mockito.doReturn;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import org.opensearch.ml.action.agenticindex.IndexProfile.FieldKind;
import org.opensearch.search.aggregations.Aggregations;
import org.opensearch.search.aggregations.bucket.terms.Terms;
import org.opensearch.search.aggregations.metrics.Max;
import org.opensearch.search.aggregations.metrics.Min;
import org.opensearch.search.builder.SearchSourceBuilder;
import org.opensearch.test.OpenSearchTestCase;

public class IndexProfileTests extends OpenSearchTestCase {

    static final Map<String, Object> PROPERTIES = Map
        .of(
            "product",
            Map.of("type", "keyword"),
            "title",
            Map.of("type", "text", "fields", Map.of("keyword", Map.of("type", "keyword"))),
            "body",
            Map.of("type", "text"),
            "published_at",
            Map.of("type", "date"),
            "page",
            Map.of("type", "integer"),
            "embedding",
            Map.of("type", "knn_vector", "dimension", 384),
            "meta",
            Map.of("properties", Map.of("archived", Map.of("type", "boolean"), "blob", Map.of("type", "binary")))
        );

    public static IndexProfile profile() {
        return IndexProfile.fromMappingProperties("docs", PROPERTIES);
    }

    public void testFieldInventory() {
        IndexProfile profile = profile();
        assertEquals(FieldKind.KEYWORD, profile.kindOf("product"));
        assertEquals(FieldKind.TEXT, profile.kindOf("title"));
        assertEquals(FieldKind.KEYWORD, profile.kindOf("title.keyword"));
        assertEquals(FieldKind.DATE, profile.kindOf("published_at"));
        assertEquals(FieldKind.NUMBER, profile.kindOf("page"));
        assertEquals(FieldKind.VECTOR, profile.kindOf("embedding"));
        assertEquals(FieldKind.BOOLEAN, profile.kindOf("meta.archived"));
        assertNull(profile.kindOf("meta.blob"));
        assertEquals(Set.of("title", "body"), new HashSet<>(profile.fieldsOfKind(FieldKind.TEXT)));
    }

    public void testStatsQueryAggregatesKeywordsAndDates() {
        SearchSourceBuilder source = profile().statsQuery();
        assertEquals(0, source.size());
        List<String> names = source.aggregations().getAggregatorFactories().stream().map(a -> a.getName()).toList();
        assertTrue(names.contains("terms__product"));
        assertTrue(names.contains("terms__title.keyword"));
        assertTrue(names.contains("min__published_at"));
        assertTrue(names.contains("max__published_at"));
        assertFalse(names.contains("terms__body"));
    }

    public void testRenderWithStatsAndInsight() {
        IndexProfile profile = profile();
        Terms terms = mock(Terms.class);
        Terms.Bucket bucket = mock(Terms.Bucket.class);
        when(terms.getName()).thenReturn("terms__product");
        when(bucket.getKeyAsString()).thenReturn("Nimbus Sync");
        doReturn(List.of(bucket)).when(terms).getBuckets();
        Min min = mock(Min.class);
        Max max = mock(Max.class);
        when(min.getName()).thenReturn("min__published_at");
        when(max.getName()).thenReturn("max__published_at");
        when(min.getValue()).thenReturn(1.0);
        when(max.getValue()).thenReturn(2.0);
        when(min.getValueAsString()).thenReturn("2026-01-01");
        when(max.getValueAsString()).thenReturn("2026-11-01");
        Aggregations aggregations = new Aggregations(List.of(terms, min, max));
        profile.applyStats(aggregations);

        String rendered = profile.withIndexInsight("product is the product line").render();
        assertTrue(rendered, rendered.contains("- product values: Nimbus Sync\n"));
        assertTrue(rendered, rendered.contains("- published_at (from 2026-01-01 to 2026-11-01)"));
        assertTrue(rendered, rendered.contains("Numeric fields: page"));
        assertTrue(rendered, rendered.contains("Index insight:\nproduct is the product line"));
        assertFalse(rendered, rendered.contains("embedding"));
    }

    public void testApplyStatsNullIsNoop() {
        IndexProfile profile = profile();
        profile.applyStats(null);
        assertTrue(profile.getKeywordValues().isEmpty());
    }

    public void testStatsQuerySkipsFieldsWithoutDocValues() {
        IndexProfile profile = IndexProfile
            .fromMappingProperties(
                "docs",
                Map
                    .of(
                        "body",
                        Map.of("type", "text", "fields", Map.of("raw", Map.of("type", "keyword", "doc_values", false))),
                        "code",
                        Map.of("type", "keyword", "doc_values", false),
                        "plan",
                        Map.of("type", "keyword"),
                        "seen_at",
                        Map.of("type", "date", "doc_values", "false")
                    )
            );
        List<String> names = profile.statsQuery().aggregations().getAggregatorFactories().stream().map(a -> a.getName()).toList();
        assertEquals(List.of("terms__plan"), names);
        assertFalse(profile.hasDocValues("code"));
        assertFalse(profile.hasDocValues("body.raw"));
        assertTrue(profile.hasDocValues("plan"));
        // Fields without doc values are still listed as filters.
        assertTrue(profile.render().contains("- code\n"));
    }

    public void testProfiledFieldsAreCappedAndPreferTextFields() {
        Map<String, Object> properties = new LinkedHashMap<>();
        for (int i = 0; i < IndexProfile.MAX_PROFILED_FIELDS + 10; i++) {
            properties.put(String.format(Locale.ROOT, "num_%02d", i), Map.of("type", "integer"));
        }
        properties.put("title", Map.of("type", "text", "fields", Map.of("keyword", Map.of("type", "keyword"))));
        properties.put("notes", Map.of("type", "text"));
        properties.put("plan", Map.of("type", "keyword"));

        IndexProfile profile = IndexProfile.fromMappingProperties("docs", properties, List.of("notes"));

        List<String> profiled = profile.getProfiledFields();
        assertEquals(IndexProfile.MAX_PROFILED_FIELDS, profiled.size());
        assertEquals(List.of("notes", "title", "plan", "num_00"), profiled.subList(0, 4));
        assertFalse("keyword sub-fields of text come last", profiled.contains("title.keyword"));
        assertEquals(FieldKind.KEYWORD, profile.kindOf("title.keyword"));
        assertTrue(profile.render().contains("(14 more fields not profiled)"));
        assertEquals(1, profile.statsQuery().aggregations().getAggregatorFactories().size());
    }

    public void testValuesAndRenderAreTruncated() {
        IndexProfile profile = IndexProfile.fromMappingProperties("docs", Map.of("product", Map.of("type", "keyword")));
        Terms terms = mock(Terms.class);
        Terms.Bucket bucket = mock(Terms.Bucket.class);
        when(terms.getName()).thenReturn("terms__product");
        when(bucket.getKeyAsString()).thenReturn("x".repeat(500));
        doReturn(List.of(bucket)).when(terms).getBuckets();
        profile.applyStats(new Aggregations(List.of(terms)));
        assertEquals(IndexProfile.MAX_VALUE_CHARS + 3, profile.getKeywordValues().get("product").get(0).length());

        String rendered = profile.withIndexInsight("y".repeat(IndexProfile.MAX_RENDERED_CHARS)).render();
        assertTrue(rendered.endsWith("... (profile truncated)\n"));
        assertTrue(rendered.length() < IndexProfile.MAX_RENDERED_CHARS + 30);
    }
}
