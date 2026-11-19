/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.agenticindex;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.junit.Test;
import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.common.io.stream.BytesStreamOutput;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.common.xcontent.XContentHelper;
import org.opensearch.common.xcontent.json.JsonXContent;
import org.opensearch.core.common.bytes.BytesReference;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.tasks.TaskId;
import org.opensearch.core.xcontent.DeprecationHandler;
import org.opensearch.core.xcontent.MediaTypeRegistry;
import org.opensearch.core.xcontent.NamedXContentRegistry;
import org.opensearch.core.xcontent.ToXContent;
import org.opensearch.core.xcontent.XContentBuilder;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.ml.common.transport.agenticindex.MLAgenticIndexRequest.Mode;
import org.opensearch.tasks.CancellableTask;
import org.opensearch.tasks.Task;

public class MLAgenticIndexRequestTest {

    private MLAgenticIndexRequest parse(String json) throws IOException {
        return parse(json, "docs");
    }

    private MLAgenticIndexRequest parse(String json, String pathIndex) throws IOException {
        XContentParser parser = JsonXContent.jsonXContent
            .createParser(NamedXContentRegistry.EMPTY, DeprecationHandler.THROW_UNSUPPORTED_OPERATION, json);
        parser.nextToken();
        return MLAgenticIndexRequest.parse(parser, pathIndex, Mode.ASK, "tenant");
    }

    @Test
    public void parse_allFields() throws IOException {
        MLAgenticIndexRequest request = parse(
            "{\"question\":\"q\",\"llm_model_id\":\"llm\",\"embedding_model_id\":\"emb\",\"embedding_field\":\"vec\","
                + "\"text_fields\":[\"title\",\"body\"],\"max_rounds\":2,\"max_subqueries\":3,\"k\":7,\"use_index_insight\":true,"
                + "\"timeout\":\"30s\",\"max_indices_selected\":7}"
        );
        assertEquals("docs", request.getIndex());
        assertEquals(Mode.ASK, request.getMode());
        assertEquals("q", request.getQuestion());
        assertEquals("llm", request.getLlmModelId());
        assertEquals("emb", request.getEmbeddingModelId());
        assertEquals("vec", request.getEmbeddingField());
        assertEquals(List.of("title", "body"), request.getTextFields());
        assertEquals(2, request.getMaxRounds());
        assertEquals(3, request.getMaxSubqueries());
        assertEquals(7, request.getK());
        assertTrue(request.isUseIndexInsight());
        assertEquals(TimeValue.timeValueSeconds(30), request.getTimeout());
        assertEquals("tenant", request.getTenantId());
        assertEquals(7, request.getMaxIndicesSelected());
        assertNull(request.validate());
    }

    @Test
    public void parse_defaults() throws IOException {
        MLAgenticIndexRequest request = parse("{\"question\":\"q\",\"llm_model_id\":\"llm\"}");
        assertEquals(MLAgenticIndexRequest.DEFAULT_MAX_ROUNDS, request.getMaxRounds());
        assertEquals(MLAgenticIndexRequest.DEFAULT_MAX_SUBQUERIES, request.getMaxSubqueries());
        assertEquals(MLAgenticIndexRequest.DEFAULT_K, request.getK());
        assertEquals(List.of(), request.getTextFields());
        assertFalse(request.isUseIndexInsight());
        assertNull(request.getTimeout());
        assertEquals(MLAgenticIndexRequest.DEFAULT_MAX_INDICES_SELECTED, request.getMaxIndicesSelected());
        assertNull(request.validate());
    }

    @Test
    public void parse_indicesInBody() throws IOException {
        MLAgenticIndexRequest request = parse("{\"question\":\"q\",\"llm_model_id\":\"llm\",\"indices\":[\"logs-*\",\"docs\"]}", null);
        assertEquals("logs-*,docs", request.getIndex());
        assertNull(request.validate());
    }

    @Test
    public void parse_indicesInBodyAndPath() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parse("{\"indices\":[\"a\"]}"));
        assertEquals("Set the target index in the path or in indices, not both", e.getMessage());
    }

    @Test
    public void parse_invalidIndices() {
        for (String indices : List.of("[]", "[\" \"]", "[\"a,b\"]")) {
            IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parse("{\"indices\":" + indices + "}", null));
            assertEquals("indices must be a non-empty list of index names, aliases or patterns", e.getMessage());
        }
    }

    @Test
    public void validate_missingIndex() {
        MLAgenticIndexRequest request = MLAgenticIndexRequest.builder().mode(Mode.ASK).question("q").llmModelId("m").build();
        assertEquals(List.of("index is required: set it in the path or in the indices body field"), request.validate().validationErrors());
    }

    @Test
    public void validate_embeddingModelWithoutFieldDetectsField() {
        MLAgenticIndexRequest request = MLAgenticIndexRequest
            .builder()
            .index("docs")
            .mode(Mode.ASK)
            .question("q")
            .llmModelId("m")
            .embeddingModelId("emb")
            .build();
        assertNull(request.validate());
    }

    @Test
    public void parse_unknownField() {
        IllegalArgumentException e = assertThrows(IllegalArgumentException.class, () -> parse("{\"questoin\":\"q\"}"));
        assertEquals("Unknown field [questoin] in agentic index request", e.getMessage());
    }

    @Test
    public void validate_missingAndOutOfRange() {
        MLAgenticIndexRequest request = MLAgenticIndexRequest
            .builder()
            .index("docs")
            .mode(Mode.RETRIEVE)
            .embeddingField("vec")
            .maxRounds(0)
            .k(21)
            .maxIndicesSelected(21)
            .build();
        ActionRequestValidationException e = request.validate();
        assertNotNull(e);
        List<String> errors = e.validationErrors();
        assertTrue(errors.contains("question is required"));
        assertTrue(errors.contains("llm_model_id is required"));
        assertTrue(errors.contains("embedding_field requires embedding_model_id"));
        assertTrue(errors.contains("max_indices_selected must be between 1 and 20"));
        assertTrue(errors.contains("max_rounds must be between 1 and 5"));
        assertTrue(errors.contains("k must be between 1 and 20"));
    }

    @Test
    public void streamRoundTrip() throws IOException {
        MLAgenticIndexRequest request = MLAgenticIndexRequest
            .builder()
            .index("docs")
            .mode(Mode.RETRIEVE)
            .question("q")
            .llmModelId("llm")
            .textFields(List.of("body"))
            .timeout(TimeValue.timeValueSeconds(5))
            .maxIndicesSelected(3)
            .build();
        BytesStreamOutput out = new BytesStreamOutput();
        request.writeTo(out);
        MLAgenticIndexRequest read = new MLAgenticIndexRequest(out.bytes().streamInput());
        assertEquals("docs", read.getIndex());
        assertEquals(Mode.RETRIEVE, read.getMode());
        assertEquals("q", read.getQuestion());
        assertEquals("llm", read.getLlmModelId());
        assertNull(read.getEmbeddingModelId());
        assertEquals(List.of("body"), read.getTextFields());
        assertEquals(TimeValue.timeValueSeconds(5), read.getTimeout());
        assertNull(read.getTenantId());
        assertEquals(3, read.getMaxIndicesSelected());
    }

    @Test
    public void validate_lengthsAndBlankValues() {
        MLAgenticIndexRequest request = MLAgenticIndexRequest
            .builder()
            .index("docs")
            .mode(Mode.ASK)
            .question("q".repeat(MLAgenticIndexRequest.QUESTION_MAX_CHARS + 1))
            .llmModelId("m")
            .embeddingModelId(" ")
            .embeddingField("")
            .textFields(List.of("body", " "))
            .timeout(TimeValue.ZERO)
            .build();
        List<String> errors = request.validate().validationErrors();
        assertEquals(
            List
                .of(
                    "question must be at most 4000 characters",
                    "embedding_model_id must not be blank",
                    "embedding_field must not be blank",
                    "text_fields must not contain blank field names",
                    "timeout must be positive"
                ),
            errors
        );
    }

    @Test
    public void validate_tooManyTextFields() {
        List<String> fields = new ArrayList<>();
        for (int i = 0; i <= MLAgenticIndexRequest.TEXT_FIELDS_LIMIT; i++) {
            fields.add("f" + i);
        }
        MLAgenticIndexRequest request = MLAgenticIndexRequest
            .builder()
            .index("docs")
            .mode(Mode.ASK)
            .question("q")
            .llmModelId("m")
            .textFields(fields)
            .build();
        assertEquals(List.of("text_fields must list at most 20 fields"), request.validate().validationErrors());
        fields.remove(0);
        MLAgenticIndexRequest atLimit = MLAgenticIndexRequest
            .builder()
            .index("docs")
            .mode(Mode.ASK)
            .question("q")
            .llmModelId("m")
            .textFields(fields)
            .build();
        assertNull(atLimit.validate());
    }

    @Test
    public void parse_invalidTimeout() {
        assertThrows(IllegalArgumentException.class, () -> parse("{\"question\":\"q\",\"timeout\":\"soon\"}"));
    }

    @Test
    public void createTask_isCancellable() {
        MLAgenticIndexRequest request = MLAgenticIndexRequest.builder().index("docs").mode(Mode.ASK).question("q").llmModelId("m").build();
        Task task = request.createTask(1, "transport", MLAgenticIndexAction.NAME, TaskId.EMPTY_TASK_ID, Map.of());
        assertTrue(task instanceof CancellableTask);
        assertTrue(((CancellableTask) task).shouldCancelChildrenOnCancellation());
        assertEquals("agentic index ASK on [docs]", task.getDescription());
    }

    @Test
    public void fromActionRequest_sameInstance() {
        MLAgenticIndexRequest request = MLAgenticIndexRequest.builder().index("docs").mode(Mode.ASK).question("q").llmModelId("m").build();
        assertSame(request, MLAgenticIndexRequest.fromActionRequest(request));
    }

    @Test
    public void fromActionRequest_otherType() {
        MLAgenticIndexRequest request = MLAgenticIndexRequest.builder().index("docs").mode(Mode.ASK).question("q").llmModelId("m").build();
        ActionRequest wrapped = new ActionRequest() {
            @Override
            public ActionRequestValidationException validate() {
                return null;
            }

            @Override
            public void writeTo(StreamOutput out) throws IOException {
                request.writeTo(out);
            }
        };
        assertEquals("q", MLAgenticIndexRequest.fromActionRequest(wrapped).getQuestion());
    }

    @Test
    public void responseRoundTrip() throws IOException {
        MLAgenticIndexResponse response = new MLAgenticIndexResponse(
            Map.of("answer", "a [P1]", "citations", List.of(Map.of("passage_id", "P1", "page", 3)))
        );
        BytesStreamOutput out = new BytesStreamOutput();
        response.writeTo(out);
        MLAgenticIndexResponse read = new MLAgenticIndexResponse(out.bytes().streamInput());
        assertEquals(response.getResult(), read.getResult());
        XContentBuilder builder = JsonXContent.contentBuilder();
        read.toXContent(builder, ToXContent.EMPTY_PARAMS);
        Map<String, Object> rendered = XContentHelper.convertToMap(BytesReference.bytes(builder), false, MediaTypeRegistry.JSON).v2();
        assertEquals(response.getResult(), rendered);
    }
}
