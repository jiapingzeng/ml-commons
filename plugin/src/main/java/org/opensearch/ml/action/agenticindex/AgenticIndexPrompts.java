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
