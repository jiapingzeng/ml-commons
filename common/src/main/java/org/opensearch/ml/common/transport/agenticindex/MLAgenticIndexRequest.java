/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.common.transport.agenticindex;

import static org.opensearch.action.ValidateActions.addValidationError;
import static org.opensearch.core.xcontent.XContentParserUtils.ensureExpectedToken;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;

import org.opensearch.action.ActionRequest;
import org.opensearch.action.ActionRequestValidationException;
import org.opensearch.common.unit.TimeValue;
import org.opensearch.core.common.io.stream.InputStreamStreamInput;
import org.opensearch.core.common.io.stream.OutputStreamStreamOutput;
import org.opensearch.core.common.io.stream.StreamInput;
import org.opensearch.core.common.io.stream.StreamOutput;
import org.opensearch.core.tasks.TaskId;
import org.opensearch.core.xcontent.XContentParser;
import org.opensearch.tasks.CancellableTask;
import org.opensearch.tasks.Task;

import lombok.Builder;
import lombok.Getter;
import lombok.ToString;

/**
 * Request for the experimental agentic index ask and retrieve APIs.
 */
@Getter
@ToString
public class MLAgenticIndexRequest extends ActionRequest {

    public enum Mode {
        ASK,
        RETRIEVE
    }

    public static final String QUESTION_FIELD = "question";
    public static final String LLM_MODEL_ID_FIELD = "llm_model_id";
    public static final String EMBEDDING_MODEL_ID_FIELD = "embedding_model_id";
    public static final String EMBEDDING_FIELD_FIELD = "embedding_field";
    public static final String TEXT_FIELDS_FIELD = "text_fields";
    public static final String MAX_ROUNDS_FIELD = "max_rounds";
    public static final String MAX_SUBQUERIES_FIELD = "max_subqueries";
    public static final String K_FIELD = "k";
    public static final String USE_INDEX_INSIGHT_FIELD = "use_index_insight";
    public static final String TIMEOUT_FIELD = "timeout";
    public static final String INDICES_FIELD = "indices";
    public static final String MAX_INDICES_SELECTED_FIELD = "max_indices_selected";

    public static final int DEFAULT_MAX_ROUNDS = 3;
    public static final int MAX_ROUNDS_LIMIT = 5;
    public static final int DEFAULT_MAX_SUBQUERIES = 4;
    public static final int MAX_SUBQUERIES_LIMIT = 8;
    public static final int DEFAULT_K = 5;
    public static final int K_LIMIT = 20;
    public static final int TEXT_FIELDS_LIMIT = 20;
    public static final int QUESTION_MAX_CHARS = 4000;
    public static final int DEFAULT_MAX_INDICES_SELECTED = 5;
    public static final int MAX_INDICES_SELECTED_LIMIT = 20;

    // Target expression: one index, alias, wildcard or comma-separated list.
    private final String index;
    private final Mode mode;
    private final String question;
    private final String llmModelId;
    private final String embeddingModelId;
    private final String embeddingField;
    private final List<String> textFields;
    private final int maxRounds;
    private final int maxSubqueries;
    private final int k;
    private final boolean useIndexInsight;
    // Optional per-request deadline; the effective deadline is capped by the cluster setting.
    private final TimeValue timeout;
    private final String tenantId;
    // Most indices the routing step selects for planning when the target resolves to several indices.
    private final int maxIndicesSelected;

    @Builder
    public MLAgenticIndexRequest(
        String index,
        Mode mode,
        String question,
        String llmModelId,
        String embeddingModelId,
        String embeddingField,
        List<String> textFields,
        Integer maxRounds,
        Integer maxSubqueries,
        Integer k,
        Boolean useIndexInsight,
        TimeValue timeout,
        String tenantId,
        Integer maxIndicesSelected
    ) {
        this.index = index;
        this.mode = mode;
        this.question = question;
        this.llmModelId = llmModelId;
        this.embeddingModelId = embeddingModelId;
        this.embeddingField = embeddingField;
        this.textFields = textFields == null ? List.of() : List.copyOf(textFields);
        this.maxRounds = maxRounds == null ? DEFAULT_MAX_ROUNDS : maxRounds;
        this.maxSubqueries = maxSubqueries == null ? DEFAULT_MAX_SUBQUERIES : maxSubqueries;
        this.k = k == null ? DEFAULT_K : k;
        this.useIndexInsight = Boolean.TRUE.equals(useIndexInsight);
        this.timeout = timeout;
        this.tenantId = tenantId;
        this.maxIndicesSelected = maxIndicesSelected == null ? DEFAULT_MAX_INDICES_SELECTED : maxIndicesSelected;
    }

    public MLAgenticIndexRequest(StreamInput in) throws IOException {
        super(in);
        this.index = in.readString();
        this.mode = in.readEnum(Mode.class);
        this.question = in.readString();
        this.llmModelId = in.readString();
        this.embeddingModelId = in.readOptionalString();
        this.embeddingField = in.readOptionalString();
        this.textFields = in.readStringList();
        this.maxRounds = in.readVInt();
        this.maxSubqueries = in.readVInt();
        this.k = in.readVInt();
        this.useIndexInsight = in.readBoolean();
        this.timeout = in.readOptionalTimeValue();
        this.tenantId = in.readOptionalString();
        this.maxIndicesSelected = in.readVInt();
    }

    @Override
    public void writeTo(StreamOutput out) throws IOException {
        super.writeTo(out);
        out.writeString(index);
        out.writeEnum(mode);
        out.writeString(question);
        out.writeString(llmModelId);
        out.writeOptionalString(embeddingModelId);
        out.writeOptionalString(embeddingField);
        out.writeStringCollection(textFields);
        out.writeVInt(maxRounds);
        out.writeVInt(maxSubqueries);
        out.writeVInt(k);
        out.writeBoolean(useIndexInsight);
        out.writeOptionalTimeValue(timeout);
        out.writeOptionalString(tenantId);
        out.writeVInt(maxIndicesSelected);
    }

    @Override
    public ActionRequestValidationException validate() {
        ActionRequestValidationException exception = null;
        if (isBlank(index)) {
            exception = addValidationError("index is required: set it in the path or in the " + INDICES_FIELD + " body field", exception);
        }
        if (mode == null) {
            exception = addValidationError("mode is required", exception);
        }
        if (isBlank(question)) {
            exception = addValidationError(QUESTION_FIELD + " is required", exception);
        } else if (question.length() > QUESTION_MAX_CHARS) {
            exception = addValidationError(
                String.format(Locale.ROOT, "%s must be at most %d characters", QUESTION_FIELD, QUESTION_MAX_CHARS),
                exception
            );
        }
        if (isBlank(llmModelId)) {
            exception = addValidationError(LLM_MODEL_ID_FIELD + " is required", exception);
        }
        // Without embedding_field, the vector field is detected per index.
        if (embeddingField != null && embeddingModelId == null) {
            exception = addValidationError(EMBEDDING_FIELD_FIELD + " requires " + EMBEDDING_MODEL_ID_FIELD, exception);
        }
        if (embeddingModelId != null && embeddingModelId.isBlank()) {
            exception = addValidationError(EMBEDDING_MODEL_ID_FIELD + " must not be blank", exception);
        }
        if (embeddingField != null && embeddingField.isBlank()) {
            exception = addValidationError(EMBEDDING_FIELD_FIELD + " must not be blank", exception);
        }
        if (textFields.stream().anyMatch(MLAgenticIndexRequest::isBlank)) {
            exception = addValidationError(TEXT_FIELDS_FIELD + " must not contain blank field names", exception);
        }
        if (textFields.size() > TEXT_FIELDS_LIMIT) {
            exception = addValidationError(
                String.format(Locale.ROOT, "%s must list at most %d fields", TEXT_FIELDS_FIELD, TEXT_FIELDS_LIMIT),
                exception
            );
        }
        if (timeout != null && timeout.nanos() <= 0) {
            exception = addValidationError(TIMEOUT_FIELD + " must be positive", exception);
        }
        exception = checkRange(MAX_ROUNDS_FIELD, maxRounds, MAX_ROUNDS_LIMIT, exception);
        exception = checkRange(MAX_SUBQUERIES_FIELD, maxSubqueries, MAX_SUBQUERIES_LIMIT, exception);
        exception = checkRange(K_FIELD, k, K_LIMIT, exception);
        exception = checkRange(MAX_INDICES_SELECTED_FIELD, maxIndicesSelected, MAX_INDICES_SELECTED_LIMIT, exception);
        return exception;
    }

    private static ActionRequestValidationException checkRange(
        String name,
        int value,
        int limit,
        ActionRequestValidationException exception
    ) {
        if (value < 1 || value > limit) {
            return addValidationError(String.format(Locale.ROOT, "%s must be between 1 and %d", name, limit), exception);
        }
        return exception;
    }

    private static boolean isBlank(String value) {
        return value == null || value.isBlank();
    }

    /**
     * Parses the request body. Unknown fields are rejected so that typos fail fast.
     * @param index target from the path, or null when the body lists the target in "indices"
     */
    public static MLAgenticIndexRequest parse(XContentParser parser, String index, Mode mode, String tenantId) throws IOException {
        MLAgenticIndexRequestBuilder builder = MLAgenticIndexRequest.builder().index(index).mode(mode).tenantId(tenantId);
        List<String> indices = null;
        ensureExpectedToken(XContentParser.Token.START_OBJECT, parser.currentToken(), parser);
        while (parser.nextToken() != XContentParser.Token.END_OBJECT) {
            String fieldName = parser.currentName();
            parser.nextToken();
            switch (fieldName) {
                case QUESTION_FIELD -> builder.question(parser.text());
                case LLM_MODEL_ID_FIELD -> builder.llmModelId(parser.text());
                case EMBEDDING_MODEL_ID_FIELD -> builder.embeddingModelId(parser.text());
                case EMBEDDING_FIELD_FIELD -> builder.embeddingField(parser.text());
                case TEXT_FIELDS_FIELD -> builder.textFields(parseStringArray(parser));
                case INDICES_FIELD -> indices = parseStringArray(parser);
                case MAX_INDICES_SELECTED_FIELD -> builder.maxIndicesSelected(parser.intValue());
                case MAX_ROUNDS_FIELD -> builder.maxRounds(parser.intValue());
                case MAX_SUBQUERIES_FIELD -> builder.maxSubqueries(parser.intValue());
                case K_FIELD -> builder.k(parser.intValue());
                case USE_INDEX_INSIGHT_FIELD -> builder.useIndexInsight(parser.booleanValue());
                case TIMEOUT_FIELD -> builder.timeout(TimeValue.parseTimeValue(parser.text(), TIMEOUT_FIELD));
                default -> throw new IllegalArgumentException("Unknown field [" + fieldName + "] in agentic index request");
            }
        }
        if (indices != null) {
            if (index != null) {
                throw new IllegalArgumentException("Set the target index in the path or in " + INDICES_FIELD + ", not both");
            }
            if (indices.isEmpty() || indices.stream().anyMatch(i -> isBlank(i) || i.contains(","))) {
                throw new IllegalArgumentException(INDICES_FIELD + " must be a non-empty list of index names, aliases or patterns");
            }
            builder.index(String.join(",", indices));
        }
        return builder.build();
    }

    private static List<String> parseStringArray(XContentParser parser) throws IOException {
        ensureExpectedToken(XContentParser.Token.START_ARRAY, parser.currentToken(), parser);
        List<String> values = new ArrayList<>();
        while (parser.nextToken() != XContentParser.Token.END_ARRAY) {
            values.add(parser.text());
        }
        return values;
    }

    @Override
    public Task createTask(long id, String type, String action, TaskId parentTaskId, Map<String, String> headers) {
        return new CancellableTask(id, type, action, "agentic index " + mode + " on [" + index + "]", parentTaskId, headers) {
            @Override
            public boolean shouldCancelChildrenOnCancellation() {
                return true;
            }
        };
    }

    public static MLAgenticIndexRequest fromActionRequest(ActionRequest actionRequest) {
        if (actionRequest instanceof MLAgenticIndexRequest) {
            return (MLAgenticIndexRequest) actionRequest;
        }
        try (ByteArrayOutputStream baos = new ByteArrayOutputStream(); OutputStreamStreamOutput osso = new OutputStreamStreamOutput(baos)) {
            actionRequest.writeTo(osso);
            try (StreamInput input = new InputStreamStreamInput(new ByteArrayInputStream(baos.toByteArray()))) {
                return new MLAgenticIndexRequest(input);
            }
        } catch (IOException e) {
            throw new UncheckedIOException("Failed to parse ActionRequest into MLAgenticIndexRequest", e);
        }
    }
}
