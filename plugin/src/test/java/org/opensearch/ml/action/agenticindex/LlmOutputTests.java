/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import static org.mockito.Mockito.mock;

import java.util.List;
import java.util.Map;

import org.opensearch.OpenSearchStatusException;
import org.opensearch.core.rest.RestStatus;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.test.OpenSearchTestCase;

public class LlmOutputTests extends OpenSearchTestCase {

    public void testStripThinkingBlock() {
        assertEquals("{\"a\":1}", LlmOutput.stripThinking("<think>plan {not json}</think>\n{\"a\":1}"));
    }

    public void testStripThinkingLoneClosingTag() {
        assertEquals("answer", LlmOutput.stripThinking("reasoning the template opened\n</think>\n\nanswer"));
    }

    public void testStripThinkingNoThinking() {
        assertEquals("plain", LlmOutput.stripThinking("  plain \n"));
    }

    public void testExtractJsonObjectFromFencedProse() {
        Map<String, Object> json = LlmOutput.extractJsonObject("Here you go:\n```json\n{\"subqueries\":[{\"query\":\"a}b\"}]}\n```");
        assertEquals("a}b", ((Map<?, ?>) ((List<?>) json.get("subqueries")).get(0)).get("query"));
    }

    public void testExtractJsonObjectSkipsInvalidCandidate() {
        Map<String, Object> json = LlmOutput.extractJsonObject("{not json} then {\"sufficient\": true}");
        assertEquals(Boolean.TRUE, json.get("sufficient"));
    }

    public void testExtractJsonObjectNone() {
        IllegalArgumentException e = expectThrows(IllegalArgumentException.class, () -> LlmOutput.extractJsonObject("no json {"));
        assertEquals("LLM output contains no JSON object", e.getMessage());
    }

    public void testCitedPassageIds() {
        assertEquals(List.of("P2", "P1", "P4"), LlmOutput.citedPassageIds("A [P2]. B [P1, P4]. C [P2]. Not a cite: P9 or [x]."));
    }

    public void testRelabelCitations() {
        Map<String, String> labels = Map.of("P1", "P3", "P2", "P1");
        assertEquals("A [P3]. B [P1, P3]. C. D", LlmOutput.relabelCitations("A [P1]. B [P2, P1]. C [P9]. D", labels));
        assertEquals("x [P3]", LlmOutput.relabelCitations("x [P1, P5]", labels));
    }

    public void testPassageIdList() {
        assertEquals(List.of("P1", "P3"), LlmOutput.passageIdList(List.of("P1", "[P3]", "none")));
        assertEquals(List.of(), LlmOutput.passageIdList("P1"));
    }

    public void testFromModelOutputChatCompletion() {
        Map<String, Object> data = Map
            .of(
                "choices",
                List.of(Map.of("message", Map.of("role", "assistant", "content", "<think>x</think>hi"))),
                "usage",
                Map.of("prompt_tokens", 12, "completion_tokens", 3)
            );
        LlmOutput.Completion completion = LlmOutput.fromModelOutput("m1", output(data));
        assertEquals("hi", completion.content());
        assertEquals(12, completion.promptTokens());
        assertEquals(3, completion.completionTokens());
    }

    public void testFromModelOutputResponseField() {
        LlmOutput.Completion completion = LlmOutput.fromModelOutput("m1", output(Map.of("response", "text")));
        assertEquals("text", completion.content());
        assertEquals(0, completion.promptTokens());
    }

    public void testFromModelOutputUnknownShapesAreBadRequest() {
        List<MLOutput> outputs = List
            .of(
                output(Map.of("x", 1)),
                output(Map.of("choices", List.of())),
                output(Map.of("choices", List.of("text"))),
                output(Map.of("choices", List.of(Map.of("message", "text")))),
                output(Map.of("choices", List.of(Map.of("message", Map.of("content", 5))))),
                ModelTensorOutput.builder().mlModelOutputs(List.of()).build(),
                ModelTensorOutput.builder().mlModelOutputs(List.of(ModelTensors.builder().mlModelTensors(List.of()).build())).build(),
                mock(MLOutput.class)
            );
        for (MLOutput output : outputs) {
            OpenSearchStatusException e = expectThrows(OpenSearchStatusException.class, () -> LlmOutput.fromModelOutput("m1", output));
            assertEquals(RestStatus.BAD_REQUEST, e.status());
            assertEquals(
                "Model m1 did not return a chat completion. Use a connector that returns choices[0].message.content.",
                e.getMessage()
            );
        }
    }

    public void testFromModelOutputListContentAndNullContent() {
        Map<String, Object> parts = Map
            .of(
                "choices",
                List.of(Map.of("message", Map.of("content", List.of(Map.of("type", "text", "text", "a"), Map.of("type", "image"), "b"))))
            );
        assertEquals("ab", LlmOutput.fromModelOutput("m1", output(parts)).content());
        Map<String, Object> message = new java.util.HashMap<>();
        message.put("content", null);
        assertEquals("", LlmOutput.fromModelOutput("m1", output(Map.of("choices", List.of(Map.of("message", message))))).content());
    }

    static ModelTensorOutput output(Map<String, ?> data) {
        ModelTensor tensor = ModelTensor.builder().name("response").dataAsMap(data).build();
        return ModelTensorOutput.builder().mlModelOutputs(List.of(ModelTensors.builder().mlModelTensors(List.of(tensor)).build())).build();
    }
}
