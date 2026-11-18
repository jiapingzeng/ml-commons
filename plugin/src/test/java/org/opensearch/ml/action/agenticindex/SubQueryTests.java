/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import java.io.IOException;
import java.util.List;
import java.util.Map;

import org.opensearch.index.query.BoolQueryBuilder;
import org.opensearch.index.query.MultiMatchQueryBuilder;
import org.opensearch.index.query.RangeQueryBuilder;
import org.opensearch.index.query.TermQueryBuilder;
import org.opensearch.index.query.TermsQueryBuilder;
import org.opensearch.index.query.WrapperQueryBuilder;
import org.opensearch.ml.action.agenticindex.SubQuery.SearchMode;
import org.opensearch.test.OpenSearchTestCase;

public class SubQueryTests extends OpenSearchTestCase {

    private final IndexProfile profile = IndexProfileTests.profile();

    public void testFromPlanValidFilters() {
        SubQuery subQuery = SubQuery
            .fromPlan(
                Map
                    .of(
                        "query",
                        " E4012 ",
                        "search",
                        "keyword",
                        "filters",
                        Map.of("product", List.of("Nimbus Sync"), "published_at", Map.of("gte", "2026-10-01", "bogus", "x"), "page", 3.0)
                    ),
                profile,
                true
            );
        assertEquals("E4012", subQuery.getText());
        assertEquals(SearchMode.KEYWORD, subQuery.getMode());
        assertEquals("Nimbus Sync", subQuery.getFilters().get("product"));
        assertEquals(Map.of("gte", "2026-10-01"), subQuery.getFilters().get("published_at"));
        assertEquals(3L, subQuery.getFilters().get("page"));
        assertTrue(subQuery.getWarnings().isEmpty());
    }

    public void testFromPlanDropsInvalidFilters() {
        SubQuery subQuery = SubQuery
            .fromPlan(
                Map
                    .of(
                        "query",
                        "sso",
                        "filters",
                        Map.of("body", "x", "unknown", "y", "embedding", "z", "product", Map.of("gte", "a"), "title", "Pricing")
                    ),
                profile,
                false
            );
        assertEquals(Map.of("title.keyword", "Pricing"), subQuery.getFilters());
        assertEquals(SearchMode.KEYWORD, subQuery.getMode());
        assertEquals(4, subQuery.getWarnings().size());
    }

    public void testFromPlanModeHandling() {
        assertEquals(SearchMode.HYBRID, SubQuery.fromPlan(Map.of("query", "a"), profile, true).getMode());
        assertEquals(SearchMode.SEMANTIC, SubQuery.fromPlan(Map.of("query", "a", "search", "Semantic"), profile, true).getMode());
        SubQuery fallback = SubQuery.fromPlan(Map.of("query", "a", "search", "semantic"), profile, false);
        assertEquals(SearchMode.KEYWORD, fallback.getMode());
        assertEquals(List.of("semantic search not configured, used keyword"), fallback.getWarnings());
        SubQuery unknown = SubQuery.fromPlan(Map.of("query", "a", "search", "fuzzy"), profile, true);
        assertEquals(SearchMode.KEYWORD, unknown.getMode());
    }

    public void testFromPlanRejectsEmptyAndNonObjects() {
        assertNull(SubQuery.fromPlan("text", profile, true));
        assertNull(SubQuery.fromPlan(Map.of("query", " ", "filters", Map.of("unknown", 1)), profile, true));
        assertNull(SubQuery.fromPlan(Map.of("query", "", "filters", Map.of("product", List.of())), profile, true));
        SubQuery badFilters = SubQuery.fromPlan(Map.of("query", "a", "filters", "product:x"), profile, true);
        assertEquals(List.of("ignored filters that are not an object"), badFilters.getWarnings());
    }

    public void testKeywordQuery() {
        SubQuery subQuery = SubQuery
            .fromPlan(
                Map
                    .of(
                        "query",
                        "error E4012",
                        "filters",
                        Map.of("product", List.of("A", "B"), "published_at", Map.of("gt", "2026-01-01", "lt", "2026-02-01"))
                    ),
                profile,
                false
            );
        BoolQueryBuilder bool = (BoolQueryBuilder) subQuery.keywordQuery(List.of("title", "body"));
        assertEquals(1, bool.must().size());
        MultiMatchQueryBuilder match = (MultiMatchQueryBuilder) bool.must().get(0);
        assertEquals("error E4012", match.value());
        assertEquals(2, match.fields().size());
        assertEquals(2, bool.filter().size());
        assertEquals(1, bool.filter().stream().filter(q -> q instanceof TermsQueryBuilder).count());
        RangeQueryBuilder range = (RangeQueryBuilder) bool.filter().stream().filter(q -> q instanceof RangeQueryBuilder).findFirst().get();
        assertEquals("2026-01-01", range.from());
        assertEquals("2026-02-01", range.to());
        assertFalse(range.includeLower());
    }

    public void testFilterOnlyKeywordQuery() {
        SubQuery subQuery = SubQuery.fromPlan(Map.of("filters", Map.of("product", "A")), profile, false);
        BoolQueryBuilder bool = (BoolQueryBuilder) subQuery.keywordQuery(List.of("body"));
        assertTrue(bool.must().isEmpty());
        assertTrue(bool.filter().get(0) instanceof TermQueryBuilder);
    }

    public void testSemanticQueryEmbedsFilter() throws IOException {
        SubQuery subQuery = SubQuery.fromPlan(Map.of("query", "single sign-on", "filters", Map.of("product", "A")), profile, true);
        WrapperQueryBuilder wrapper = (WrapperQueryBuilder) subQuery.semanticQuery("embedding", "model-1", 5);
        String json = new String(wrapper.source(), java.nio.charset.StandardCharsets.UTF_8);
        assertTrue(json, json.startsWith("{\"neural\":{\"embedding\":{\"query_text\":\"single sign-on\",\"model_id\":\"model-1\",\"k\":5"));
        assertTrue(json, json.contains("\"filter\":{\"bool\":{\"filter\":[{\"term\":{\"product\""));
    }

    public void testToMap() {
        SubQuery subQuery = SubQuery.fromPlan(Map.of("query", "a", "search", "keyword", "filters", Map.of("unknown", 1)), profile, true);
        Map<String, Object> map = subQuery.toMap();
        assertEquals("a", map.get("query"));
        assertEquals("keyword", map.get("search"));
        assertEquals(Map.of(), map.get("filters"));
        assertEquals(List.of("dropped filter on non-filterable field unknown"), map.get("warnings"));
    }
}
