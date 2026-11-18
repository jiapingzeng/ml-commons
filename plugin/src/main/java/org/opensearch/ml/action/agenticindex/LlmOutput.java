/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import static org.opensearch.ml.common.utils.StringUtils.gson;

import java.util.ArrayList;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;

import com.google.gson.JsonParseException;

/**
 * Parsing helpers for chat-completion output: thinking removal, JSON extraction and citation markers.
 */
public final class LlmOutput {

    private static final Pattern THINK_BLOCK = Pattern.compile("(?s)<think>.*?</think>");
    private static final Pattern CITATION_GROUP = Pattern.compile("\\[(P\\d+(?:\\s*,\\s*P\\d+)*)\\]");
    private static final Pattern PASSAGE_ID = Pattern.compile("P\\d+");

    private LlmOutput() {}

    /**
     * Chat completion text and token usage of one LLM call.
     */
    public record Completion(String content, long promptTokens, long completionTokens) {
    }

    /**
     * Reads an OpenAI-style chat completion (choices[0].message.content, usage) from the raw connector output.
     * Connectors with a post-process function that yields a "response" string are accepted as well. Content given as a
     * list of parts is joined from its text parts.
     *
     * @throws OpenSearchStatusException with status 400 when the output has neither shape
     */
    @SuppressWarnings("unchecked")
    public static Completion fromModelOutput(String modelId, MLOutput output) {
        Map<String, ?> data = firstTensorData(output);
        String content = null;
        if (data != null
            && data.get("choices") instanceof List<?> choices
            && !choices.isEmpty()
            && choices.get(0) instanceof Map<?, ?> choice
            && choice.get("message") instanceof Map<?, ?> message) {
            // Null content is a valid completion, for example when reasoning used up max_tokens.
            content = message.get("content") == null ? "" : contentText(message.get("content"));
        } else if (data != null && data.get("response") instanceof String response) {
            content = response;
        }
        if (content == null) {
            throw new OpenSearchStatusException(
                "Model " + modelId + " did not return a chat completion. Use a connector that returns choices[0].message.content.",
                RestStatus.BAD_REQUEST
            );
        }
        long promptTokens = 0;
        long completionTokens = 0;
        if (data.get("usage") instanceof Map) {
            Map<String, Object> usage = (Map<String, Object>) data.get("usage");
            promptTokens = toLong(usage.get("prompt_tokens"));
            completionTokens = toLong(usage.get("completion_tokens"));
        }
        return new Completion(stripThinking(content), promptTokens, completionTokens);
    }

    private static Map<String, ?> firstTensorData(MLOutput output) {
        if (!(output instanceof ModelTensorOutput tensorOutput)) {
            return null;
        }
        List<ModelTensors> outputs = tensorOutput.getMlModelOutputs();
        if (outputs == null || outputs.isEmpty() || outputs.get(0).getMlModelTensors() == null) {
            return null;
        }
        List<ModelTensor> tensors = outputs.get(0).getMlModelTensors();
        return tensors.isEmpty() ? null : tensors.get(0).getDataAsMap();
    }

    // Content is a string, or (in newer chat APIs) a list of parts such as {"type": "text", "text": "..."}.
    private static String contentText(Object content) {
        if (content instanceof String text) {
            return text;
        }
        if (!(content instanceof List<?> parts)) {
            return null;
        }
        StringBuilder sb = new StringBuilder();
        for (Object part : parts) {
            if (part instanceof String text) {
                sb.append(text);
            } else if (part instanceof Map<?, ?> map && map.get("text") instanceof String text) {
                sb.append(text);
            }
        }
        return sb.toString();
    }

    private static long toLong(Object value) {
        return value instanceof Number n ? n.longValue() : 0;
    }

    /**
     * Removes reasoning text. Serving stacks without a reasoning parser return it inline, either as a full
     * think block or, when the template opens the block itself, as text ending in a lone closing tag.
     */
    public static String stripThinking(String text) {
        String result = THINK_BLOCK.matcher(text).replaceAll("");
        int close = result.lastIndexOf("</think>");
        if (close >= 0) {
            result = result.substring(close + "</think>".length());
        }
        return result.trim();
    }

    /**
     * Extracts the first balanced JSON object from model text (fences and prose are ignored).
     *
     * @throws IllegalArgumentException when no parseable object is present
     */
    @SuppressWarnings("unchecked")
    public static Map<String, Object> extractJsonObject(String text) {
        int start = text.indexOf('{');
        while (start >= 0) {
            int end = matchingBrace(text, start);
            if (end < 0) {
                break;
            }
            try {
                Object parsed = gson.fromJson(text.substring(start, end + 1), Object.class);
                if (parsed instanceof Map) {
                    return (Map<String, Object>) parsed;
                }
            } catch (JsonParseException e) {
                // try the next candidate object
            }
            start = text.indexOf('{', start + 1);
        }
        throw new IllegalArgumentException("LLM output contains no JSON object");
    }

    private static int matchingBrace(String text, int start) {
        int depth = 0;
        boolean inString = false;
        for (int i = start; i < text.length(); i++) {
            char c = text.charAt(i);
            if (inString) {
                if (c == '\\') {
                    i++;
                } else if (c == '"') {
                    inString = false;
                }
            } else if (c == '"') {
                inString = true;
            } else if (c == '{') {
                depth++;
            } else if (c == '}' && --depth == 0) {
                return i;
            }
        }
        return -1;
    }

    /**
     * Passage ids cited as [P1] or [P1, P3], in first-cited order.
     */
    public static List<String> citedPassageIds(String answer) {
        Set<String> ids = new LinkedHashSet<>();
        Matcher group = CITATION_GROUP.matcher(answer);
        while (group.find()) {
            Matcher id = PASSAGE_ID.matcher(group.group(1));
            while (id.find()) {
                ids.add(id.group());
            }
        }
        return new ArrayList<>(ids);
    }

    /**
     * Rewrites citation markers through a label-to-id map, for example [P1, P2] to [P3, P1]. Labels missing from the
     * map are removed, and a marker left with no ids is removed with its leading space.
     */
    public static String relabelCitations(String answer, Map<String, String> labelToId) {
        Matcher group = Pattern.compile(" ?" + CITATION_GROUP.pattern()).matcher(answer);
        StringBuilder sb = new StringBuilder();
        while (group.find()) {
            Matcher id = PASSAGE_ID.matcher(group.group(1));
            List<String> ids = new ArrayList<>();
            while (id.find()) {
                if (labelToId.containsKey(id.group())) {
                    ids.add(labelToId.get(id.group()));
                }
            }
            String replacement = ids.isEmpty() ? "" : (group.group().startsWith(" ") ? " " : "") + "[" + String.join(", ", ids) + "]";
            group.appendReplacement(sb, Matcher.quoteReplacement(replacement));
        }
        group.appendTail(sb);
        return sb.toString();
    }

    /**
     * Normalizes a JSON value that should be a list of passage ids.
     */
    public static List<String> passageIdList(Object value) {
        List<String> ids = new ArrayList<>();
        if (value instanceof List) {
            for (Object v : (List<?>) value) {
                Matcher m = PASSAGE_ID.matcher(String.valueOf(v));
                if (m.find()) {
                    ids.add(m.group());
                }
            }
        }
        return ids;
    }
}
