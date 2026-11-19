/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;

import org.opensearch.action.OriginalIndices;
import org.opensearch.core.common.bytes.BytesArray;
import org.opensearch.core.index.shard.ShardId;
import org.opensearch.search.SearchHit;
import org.opensearch.search.SearchShardTarget;
import org.opensearch.test.OpenSearchTestCase;

public class PassageStoreTests extends OpenSearchTestCase {

    static SearchHit hit(String id, String sourceJson) {
        return hit("docs", id, sourceJson);
    }

    static SearchHit hit(String index, String id, String sourceJson) {
        SearchHit hit = new SearchHit(1, id, Map.of(), Map.of());
        hit.sourceRef(new BytesArray(sourceJson));
        hit.shard(new SearchShardTarget("node", new ShardId(index, "uuid", 0), null, OriginalIndices.NONE));
        hit.score(1.5f);
        return hit;
    }

    public void testAddDedupesAndNumbers() {
        PassageStore store = new PassageStore(List.of("title", "body"), Set.of("embedding"), 10);
        assertEquals("P1", store.add(hit("a", "{\"title\":\"T\",\"body\":\"B\",\"embedding\":[1,2]}"), 1, 0));
        assertEquals("P2", store.add(hit("b", "{\"body\":\"other\"}"), 1, 1));
        assertEquals("P1", store.add(hit("a", "{\"title\":\"T\"}"), 2, 0));
        assertEquals(2, store.size());
        Map<String, Object> passage = store.get("P1");
        assertEquals("docs", passage.get("index"));
        assertEquals("a", passage.get("doc_id"));
        assertEquals("T\nB", passage.get("text"));
        assertEquals("T", passage.get("title"));
        assertEquals(Map.of(), passage.get("metadata"));
    }

    public void testAddStopsAtLimit() {
        PassageStore store = new PassageStore(List.of("body"), Set.of(), 1);
        assertEquals("P1", store.add(hit("a", "{\"body\":\"x\"}"), 1, 0));
        assertNull(store.add(hit("b", "{\"body\":\"y\"}"), 1, 0));
    }

    public void testTruncatesLongText() {
        PassageStore store = new PassageStore(List.of("body"), Set.of(), 5);
        store.add(hit("a", "{\"body\":\"" + "x".repeat(PassageStore.MAX_TEXT_CHARS + 10) + "\"}"), 1, 0);
        assertEquals(PassageStore.MAX_TEXT_CHARS + 3, ((String) store.get("P1").get("text")).length());
    }

    public void testCitationUsesLocationFields() {
        PassageStore store = new PassageStore(List.of("title", "body"), Set.of(), 5);
        store.add(hit("a", "{\"title\":\"Pricing\",\"body\":\"b\",\"source\":\"pricing.pdf\",\"page\":3,\"plan\":[\"Pro\"]}"), 1, 0);
        Map<String, Object> citation = store.citation("P1");
        assertEquals("P1", citation.get("passage_id"));
        assertEquals("a", citation.get("doc_id"));
        assertEquals("pricing.pdf", citation.get("source"));
        assertEquals(3, citation.get("page"));
        assertEquals("Pricing", citation.get("title"));
        assertFalse(citation.containsKey("plan"));
    }

    public void testRender() {
        PassageStore store = new PassageStore(List.of("body"), Set.of(), 5);
        store.add(hit("a", "{\"body\":\"hello\",\"plan\":\"Pro\"}"), 1, 0);
        assertEquals("[P1] doc_id=a plan=Pro\nhello", store.render(store.all()));
    }

    public void testPerIndexFieldsAndIndexInRender() {
        Map<String, List<String>> textFields = Map.of("docs-a", List.of("body"), "docs-b", List.of("summary"));
        PassageStore store = new PassageStore(textFields::get, index -> Set.of("vec"), 5, true);
        store.add(hit("docs-a", "a", "{\"body\":\"pricing\",\"summary\":\"not text here\"}"), 1, 0);
        store.add(hit("docs-b", "a", "{\"summary\":\"outage\",\"vec\":[1]}"), 1, 1);
        assertEquals(2, store.size());
        assertEquals(
            "[P1] index=docs-a doc_id=a summary=not text here\npricing\n\n[P2] index=docs-b doc_id=a\noutage",
            store.render(store.all())
        );
        assertEquals("docs-b", store.citation("P2").get("index"));
    }

    public void testObjectPathTextFieldsAndParentExcludedFromMetadata() {
        PassageStore store = new PassageStore(List.of("content.body", "chunks.text"), Set.of("vectors.embedding"), 5);
        store
            .add(
                hit(
                    "a",
                    "{\"content\":{\"body\":\"SSO in Pro\",\"lang\":\"en\"},\"chunks\":[{\"text\":\"c1\"},{\"text\":\"c2\"}],"
                        + "\"vectors\":{\"model\":\"m\"},\"page\":3}"
                ),
                1,
                0
            );
        Map<String, Object> passage = store.get("P1");
        assertEquals("SSO in Pro\nc1\nc2", passage.get("text"));
        assertEquals(Map.of("page", 3), passage.get("metadata"));
    }

    public void testCapsMetadataOfWideDocument() {
        PassageStore store = new PassageStore(List.of("body"), Set.of(), 5);
        StringBuilder narrow = new StringBuilder("{\"body\":\"b\"");
        StringBuilder wide = new StringBuilder("{\"body\":\"b\"");
        for (int i = 0; i < 50; i++) {
            narrow.append(String.format(Locale.ROOT, ",\"f%02d\":\"%s\"", i, "v".repeat(50)));
            wide.append(String.format(Locale.ROOT, ",\"f%02d\":\"%s\"", i, "v".repeat(500)));
        }
        // Location fields come last in the source but are kept first.
        narrow.append(",\"source\":\"a.pdf\",\"page\":3}");
        wide.append(",\"source\":\"a.pdf\",\"page\":3}");
        store.add(hit("a", narrow.toString()), 1, 0);
        store.add(hit("b", wide.toString()), 1, 0);

        Map<String, Object> byEntries = metadata(store, "P1");
        assertEquals(PassageStore.MAX_METADATA_ENTRIES, byEntries.size());
        assertEquals(List.of("source", "page", "f00"), List.copyOf(byEntries.keySet()).subList(0, 3));

        Map<String, Object> byChars = metadata(store, "P2");
        assertEquals(List.of("source", "page", "f00", "f01", "f02", "f03"), List.copyOf(byChars.keySet()));
        int chars = byChars.entrySet().stream().mapToInt(e -> e.getKey().length() + String.valueOf(e.getValue()).length()).sum();
        assertTrue(String.valueOf(chars), chars <= PassageStore.MAX_METADATA_TOTAL_CHARS);
    }

    @SuppressWarnings("unchecked")
    private static Map<String, Object> metadata(PassageStore store, String passageId) {
        return (Map<String, Object>) store.get(passageId).get("metadata");
    }

    public void testCapsNonStringMetadata() {
        PassageStore store = new PassageStore(List.of("body"), Set.of(), 5);
        String longList = "[" + "\"tag\",".repeat(100) + "\"end\"]";
        store.add(hit("a", "{\"body\":\"b\",\"tags\":" + longList + ",\"small\":{\"k\":\"v\"},\"flag\":true}"), 1, 0);
        @SuppressWarnings("unchecked")
        Map<String, Object> metadata = (Map<String, Object>) store.get("P1").get("metadata");
        assertEquals(PassageStore.MAX_METADATA_CHARS + 3, ((String) metadata.get("tags")).length());
        assertEquals(Map.of("k", "v"), metadata.get("small"));
        assertEquals(true, metadata.get("flag"));
    }
}
