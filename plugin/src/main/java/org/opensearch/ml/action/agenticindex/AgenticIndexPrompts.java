/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import java.util.Locale;

/**
 * Prompts for the agentic index flow. Outputs are small JSON objects instead of Query DSL because smaller chat models
 * write DSL less reliably.
 */
final class AgenticIndexPrompts {

    private AgenticIndexPrompts() {}

    static final String SYSTEM = "You are an evidence-grounded search assistant. Answer questions only from the configured "
        + "OpenSearch index. Your pretrained knowledge is not evidence. Every factual claim and numeric value must be directly "
        + "supported by returned search evidence. If the index cannot answer the question, say what is unavailable without guessing.";

    private static final String SUBQUERY_FORMAT = """
        Each subquery is a JSON object:
        {"query": "<keywords or a short natural-language phrase to match against the text fields, or empty>",
         "search": "keyword" | "semantic" | "hybrid",
         "filters": {"<field>": <value> | [<values>] | {"gte": <v>, "lte": <v>, "gt": <v>, "lt": <v>}}}
        Rules:
        - Use filters only on the exact-match keyword, date, numeric and boolean fields listed in the index profile, and only \
        with values that appear in the profile (match case exactly). Use date ranges with ISO dates (yyyy-MM-dd).
        - Put exact identifiers such as error codes, SKUs and version numbers in "query" with "search": "keyword".
        - Use "semantic" or "hybrid" for conceptual wording when semantic search is available.%s
        - Split a question that asks about several things into one subquery per thing so they run in parallel.
        - Do not add a filter the question does not ask for.""";

    static String planner(String profile, String question, String now, int maxSubqueries, boolean semanticAvailable) {
        String semanticNote = semanticAvailable ? "" : "\n- Semantic search is not available; always use \"keyword\".";
        return String
            .format(
                Locale.ROOT,
                """
                    Current timestamp: %s

                    Index profile:
                    %s
                    ## Task
                    Plan the smallest set of independent search subqueries (at most %d) that together retrieve the evidence \
                    needed to answer the question.

                    %s

                    Question: %s

                    Return only a JSON object: {"subqueries": [ ... ]}""",
                now,
                profile,
                maxSubqueries,
                String.format(Locale.ROOT, SUBQUERY_FORMAT, semanticNote),
                question
            );
    }

    static String sufficiency(
        String profile,
        String question,
        String now,
        String passages,
        String previousQueries,
        int maxSubqueries,
        boolean semanticAvailable
    ) {
        String semanticNote = semanticAvailable ? "" : "\n- Semantic search is not available; always use \"keyword\".";
        return String
            .format(
                Locale.ROOT,
                """
                    Current timestamp: %s

                    Index profile:
                    %s
                    Question: %s

                    Subqueries already run:
                    %s

                    Retrieved passages:
                    %s

                    ## Task
                    Judge whether the retrieved passages contain enough evidence to fully answer every part of the question.
                    List the ids of the passages that directly support an answer, most relevant first; omit unrelated passages.
                    If evidence is missing, propose at most %d new follow-up subqueries that differ from the ones already run.

                    %s

                    Return only a JSON object:
                    {"sufficient": true | false, "relevant": ["P1", ...], "missing": "<what is still missing, or empty>", \
                    "followups": [ ... ]}""",
                now,
                profile,
                question,
                previousQueries,
                passages.isEmpty() ? "(none)" : passages,
                maxSubqueries,
                String.format(Locale.ROOT, SUBQUERY_FORMAT, semanticNote)
            );
    }

    private static final String MULTI_SUBQUERY_FORMAT = """
        Each subquery is a JSON object:
        {"index": "<name of one index from the profiles>",
         "query": "<keywords or a short natural-language phrase to match against that index's text fields, or empty>",
         "search": "keyword" | "semantic" | "hybrid",
         "filters": {"<field>": <value> | [<values>] | {"gte": <v>, "lte": <v>, "gt": <v>, "lt": <v>}}}
        Rules:
        - Each subquery searches one index. Use filters only on the exact-match keyword, date, numeric and boolean fields \
        listed in that index's profile, and only with values that appear there (match case exactly). Use date ranges with \
        ISO dates (yyyy-MM-dd).
        - Put exact identifiers such as error codes, SKUs and version numbers in "query" with "search": "keyword".
        - Use "semantic" or "hybrid" only on indices whose profile says semantic search is available.
        - Split a question that asks about several things, or needs several indices, into one subquery per thing and \
        index so they run in parallel.
        - Do not add a filter the question does not ask for.""";

    static String routing(String catalog, String question, String now, int maxIndices) {
        return String
            .format(
                Locale.ROOT,
                """
                    Current timestamp: %s

                    Index catalog (one line per index: fields with types):
                    %s
                    ## Task
                    Select the indices most likely to contain the evidence needed to answer the question, at most %d, most \
                    relevant first. Prefer fewer indices when one clearly covers the question. Use only index names from the catalog.

                    Question: %s

                    Return only a JSON object: {"indices": [{"index": "<name>", "reason": "<one short sentence>"}]}""",
                now,
                catalog,
                maxIndices,
                question
            );
    }

    static String multiPlanner(String profiles, String question, String now, int maxSubqueries) {
        return String.format(Locale.ROOT, """
            Current timestamp: %s

            Index profiles:
            %s
            ## Task
            Plan the smallest set of independent search subqueries (at most %d) that together retrieve the evidence \
            needed to answer the question.

            %s

            Question: %s

            Return only a JSON object: {"subqueries": [ ... ]}""", now, profiles, maxSubqueries, MULTI_SUBQUERY_FORMAT, question);
    }

    static String multiSufficiency(
        String profiles,
        String otherIndices,
        String question,
        String now,
        String passages,
        String previousQueries,
        int maxSubqueries
    ) {
        String other = otherIndices.isEmpty()
            ? ""
            : "Other indices (a follow-up subquery may search at most one of them, which adds it to the profiles):\n" + otherIndices;
        return String
            .format(
                Locale.ROOT,
                """
                    Current timestamp: %s

                    Index profiles:
                    %s
                    %s
                    Question: %s

                    Subqueries already run:
                    %s

                    Retrieved passages:
                    %s

                    ## Task
                    Judge whether the retrieved passages contain enough evidence to fully answer every part of the question.
                    List the ids of the passages that directly support an answer, most relevant first; omit unrelated passages.
                    If evidence is missing, propose at most %d new follow-up subqueries that differ from the ones already run.

                    %s

                    Return only a JSON object:
                    {"sufficient": true | false, "relevant": ["P1", ...], "missing": "<what is still missing, or empty>", \
                    "followups": [ ... ]}""",
                now,
                profiles,
                other,
                question,
                previousQueries,
                passages.isEmpty() ? "(none)" : passages,
                maxSubqueries,
                MULTI_SUBQUERY_FORMAT
            );
    }

    static String answer(String question, String now, String passages) {
        return String.format(Locale.ROOT, """
            Current timestamp: %s

            Retrieved passages:
            %s

            Question: %s

            ## Task
            Answer the question using only the passages above. Cite the supporting passage id in square brackets right \
            after every claim, for example "SSO is included in the Enterprise plan [P2]." or "[P1, P4]". Cite only \
            passages that state the claim. Answer every part of the question and include the specific values (prices, \
            limits, dates, versions) the passages give. If the passages do not answer part of the question, say so \
            explicitly. Be concise. Return only the answer text.""", now, passages, question);
    }
}
