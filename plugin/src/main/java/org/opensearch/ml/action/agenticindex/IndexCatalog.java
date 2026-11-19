/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Locale;
import java.util.Set;

/**
 * Compact catalog of many indices for the routing step, one short entry per index: the start of the mapping's
 * {@code _meta.description}, field names and types, and the keyword values and date ranges known for the index. Routing
 * runs no statistics search, so values and ranges appear only for indices whose statistics an earlier request by the
 * same caller cached. When the catalog would exceed its size budget, indices are ranked by lexical overlap of the
 * question with their names, field names and known keyword values, and only the best-ranked entries that fit are shown.
 */
final class IndexCatalog {

    static final int MAX_CHARS = 12000;
    static final int ENTRY_MAX_CHARS = 400;
    private static final Set<String> STOPWORDS = Set
        .of(
            "the",
            "and",
            "an",
            "as",
            "at",
            "be",
            "by",
            "do",
            "in",
            "is",
            "it",
            "of",
            "on",
            "or",
            "to",
            "we",
            "were",
            "been",
            "for",
            "are",
            "was",
            "what",
            "which",
            "who",
            "when",
            "where",
            "how",
            "does",
            "did",
            "with",
            "from",
            "that",
            "this",
            "there",
            "have",
            "has",
            "our",
            "any",
            "all",
            "about",
            "into",
            "between",
            "most",
            "many",
            "much",
            "show",
            "list",
            "give",
            "tell"
        );

    /** Rendered catalog and the indices it shows, in display order. */
    record Catalog(String text, List<String> indices, boolean prefiltered) {
    }

    private IndexCatalog() {}

    static Catalog render(List<IndexProfile> profiles, String question, int maxChars) {
        int total = 0;
        for (IndexProfile profile : profiles) {
            total += profile.catalogEntry(ENTRY_MAX_CHARS).length() + 1;
        }
        boolean prefiltered = total > maxChars;
        List<IndexProfile> ordered = prefiltered ? rank(profiles, question) : profiles;
        StringBuilder sb = new StringBuilder();
        List<String> shown = new ArrayList<>();
        for (IndexProfile profile : ordered) {
            String entry = profile.catalogEntry(ENTRY_MAX_CHARS);
            if (sb.length() + entry.length() + 1 > maxChars) {
                continue;
            }
            sb.append(entry).append('\n');
            shown.add(profile.getIndex());
        }
        if (shown.size() < profiles.size()) {
            sb.append(String.format(Locale.ROOT, "(%d more indices not shown)\n", profiles.size() - shown.size()));
        }
        return new Catalog(sb.toString(), shown, prefiltered);
    }

    /**
     * Profiles ordered by lexical relevance to the question, best first; ties keep their input order.
     */
    static List<IndexProfile> rank(List<IndexProfile> profiles, String question) {
        Set<String> questionTerms = terms(question);
        List<IndexProfile> ranked = new ArrayList<>(profiles);
        ranked.sort(Comparator.comparingInt((IndexProfile p) -> score(p, questionTerms)).reversed());
        return ranked;
    }

    /**
     * Question terms found in the index name count three, in field names two, and in keyword values one.
     */
    static int score(IndexProfile profile, Set<String> questionTerms) {
        Set<String> nameTerms = terms(profile.getIndex());
        Set<String> fieldTerms = new LinkedHashSet<>();
        profile.getProfiledFields().forEach(f -> fieldTerms.addAll(terms(f)));
        Set<String> valueTerms = new LinkedHashSet<>();
        profile.getKeywordValues().values().forEach(values -> values.forEach(v -> valueTerms.addAll(terms(v))));
        int score = 0;
        for (String term : questionTerms) {
            score += (nameTerms.contains(term) ? 3 : 0) + (fieldTerms.contains(term) ? 2 : 0) + (valueTerms.contains(term) ? 1 : 0);
        }
        return score;
    }

    /**
     * Lowercase alphanumeric words of at least two characters, without stopwords and with a plural "s" removed.
     */
    static Set<String> terms(String text) {
        Set<String> terms = new LinkedHashSet<>();
        for (String word : text.toLowerCase(Locale.ROOT).split("[^\\p{Alnum}]+")) {
            if (word.length() < 2 || STOPWORDS.contains(word)) {
                continue;
            }
            terms.add(word.length() > 3 && word.endsWith("s") ? word.substring(0, word.length() - 1) : word);
        }
        return terms;
    }
}
