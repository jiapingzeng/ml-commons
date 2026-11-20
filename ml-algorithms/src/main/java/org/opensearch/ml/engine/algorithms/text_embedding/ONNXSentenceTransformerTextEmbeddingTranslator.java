/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.text_embedding;

import static org.opensearch.ml.engine.algorithms.text_embedding.TextEmbeddingDenseModel.SENTENCE_EMBEDDING;

import java.io.IOException;
import java.nio.LongBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import org.opensearch.ml.common.model.TextEmbeddingModelConfig;
import org.opensearch.ml.common.output.model.MLResultDataType;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensors;

import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.translate.Batchifier;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.TranslatorContext;

/**
 * Translator for ONNX text embedding models.
 *
 * Input tensors are created directly with their final shape and pooling and normalization run in plain Java, so no
 * NDArray operation is delegated to DJL's alternative engine (PyTorch) and ONNX inference does not need libtorch.
 * The math mirrors the NDArray operations this translator used before, including their output shapes.
 */
public class ONNXSentenceTransformerTextEmbeddingTranslator implements ServingTranslator {
    private static final float MAX_POOLING_PADDING_VALUE = -1e9f;
    private static final float MEAN_POOLING_MIN_MASK_SUM = 1e-9f;
    private static final float MEAN_POOLING_MAX_MASK_SUM = 1e12f;
    private static final float NORMALIZE_EPSILON = 1e-12f;

    private HuggingFaceTokenizer tokenizer;
    private TextEmbeddingModelConfig.PoolingMode poolingMode;
    private boolean normalizeResult;
    private String modelType;

    public ONNXSentenceTransformerTextEmbeddingTranslator(
        TextEmbeddingModelConfig.PoolingMode poolingMode,
        boolean normalizeResult,
        String modelType
    ) {
        this.poolingMode = poolingMode == null ? TextEmbeddingModelConfig.PoolingMode.MEAN : poolingMode;
        this.normalizeResult = normalizeResult;
        this.modelType = modelType;
    }

    @Override
    public Batchifier getBatchifier() {
        return null;

    }

    @Override
    public void prepare(TranslatorContext ctx) throws IOException {
        Path path = ctx.getModel().getModelPath();
        tokenizer = HuggingFaceTokenizer
            .builder()
            .optPadding(true)
            .optTokenizerPath(path.resolve("tokenizer.json"))
            .optManager(ctx.getPredictorManager())
            .build();
    }

    @Override
    public NDList processInput(TranslatorContext ctx, Input input) {
        NDManager manager = ctx.getNDManager();
        String sentence = input.getAsString(0);
        NDList ndList = new NDList();

        Encoding encode = tokenizer.encode(sentence);
        ctx.setAttachment("encoding", encode);
        long[] indices = encode.getIds();
        Shape shape = new Shape(1, indices.length);

        ndList.add(createInput(manager, indices, shape, "input_ids"));
        ndList.add(createInput(manager, encode.getAttentionMask(), shape, "attention_mask"));
        if ("bert".equalsIgnoreCase(modelType) || "albert".equalsIgnoreCase(modelType)) {
            ndList.add(createInput(manager, encode.getTypeIds(), shape, "token_type_ids"));
        }
        return ndList;
    }

    private static NDArray createInput(NDManager manager, long[] data, Shape shape, String name) {
        NDArray array = manager.create(LongBuffer.wrap(data), shape);
        array.setName(name);
        return array;
    }

    /** {@inheritDoc} */
    @Override
    public Output processOutput(TranslatorContext ctx, NDList list) {
        // NONE pooling mode uses the pre-pooled second output (sentence_embedding) if available
        NDArray embeddings = this.poolingMode == TextEmbeddingModelConfig.PoolingMode.NONE && list.size() > 1 ? list.get(1) : list.get(0);
        long[] shape = embeddings.getShape().getShape();
        float[] values = toFloatArray(embeddings);
        if (shape.length == 3) {
            // keep the first (and only) item of the batch
            shape = new long[] { shape[1], shape[2] };
            values = Arrays.copyOf(values, Math.multiplyExact((int) shape[0], (int) shape[1]));
        }
        Encoding encoding = (Encoding) ctx.getAttachment("encoding");
        long[] attentionMask = encoding.getAttentionMask();

        Embedding result;
        switch (this.poolingMode) {
            case MEAN:
                result = meanPool(values, shape, attentionMask, false);
                break;
            case MEAN_SQRT_LEN:
                result = meanPool(values, shape, attentionMask, true);
                break;
            case MAX:
                result = maxPool(values, shape, attentionMask);
                break;
            case WEIGHTED_MEAN:
                result = weightedMeanPool(values, shape, attentionMask);
                break;
            case CLS:
                result = row(values, shape, 0);
                break;
            case LAST_TOKEN:
                result = row(values, shape, findLastRealTokenIndex(attentionMask));
                break;
            case NONE:
                result = new Embedding(values, shape);
                break;
            default:
                throw new IllegalArgumentException("Unsupported pooling method");
        }

        if (normalizeResult) {
            normalize(result);
        }

        Number[] data = new Number[result.values.length];
        for (int i = 0; i < data.length; i++) {
            data[i] = result.values[i];
        }
        List<ModelTensor> outputs = new ArrayList<>();
        ModelTensor modelTensor = ModelTensor
            .builder()
            .name(SENTENCE_EMBEDDING)
            .data(data)
            .shape(result.shape)
            .dataType(MLResultDataType.FLOAT32)
            .build();
        outputs.add(modelTensor);

        Output output = new Output();
        ModelTensors modelTensorOutput = new ModelTensors(outputs);
        output.add(modelTensorOutput.toBytes());
        return output;
    }

    private static float[] toFloatArray(NDArray array) {
        DataType dataType = array.getDataType();
        switch (dataType) {
            case FLOAT16:
            case FLOAT32:
                return array.toFloatArray();
            case FLOAT64:
                double[] doubles = array.toDoubleArray();
                float[] floats = new float[doubles.length];
                for (int i = 0; i < doubles.length; i++) {
                    floats[i] = (float) doubles[i];
                }
                return floats;
            default:
                throw new IllegalArgumentException(
                    "Unsupported ONNX model output data type " + dataType + ", expected float16, float32 or float64"
                );
        }
    }

    /** A dense float tensor in row-major order. */
    private static final class Embedding {
        private final float[] values;
        private final long[] shape;

        private Embedding(float[] values, long[] shape) {
            this.values = values;
            this.shape = shape;
        }
    }

    /** Returns the hidden size of a [sequence_length, hidden_size] token embedding matrix. */
    private int hiddenSize(long[] shape, long[] attentionMask) {
        if (shape.length != 2 || shape[0] != attentionMask.length) {
            throw new IllegalArgumentException(
                "Unsupported ONNX model output shape "
                    + Arrays.toString(shape)
                    + " for pooling mode "
                    + poolingMode
                    + ", expected [sequence_length, hidden_size] with sequence_length "
                    + attentionMask.length
            );
        }
        return (int) shape[1];
    }

    private Embedding meanPool(float[] values, long[] shape, long[] attentionMask, boolean sqrt) {
        int hidden = hiddenSize(shape, attentionMask);
        float[] sum = new float[hidden];
        long maskSum = 0;
        for (int i = 0; i < attentionMask.length; i++) {
            float mask = attentionMask[i];
            maskSum += attentionMask[i];
            int offset = i * hidden;
            for (int j = 0; j < hidden; j++) {
                sum[j] += values[offset + j] * mask;
            }
        }
        float clamp = Math.min(Math.max((float) maskSum, MEAN_POOLING_MIN_MASK_SUM), MEAN_POOLING_MAX_MASK_SUM);
        float divisor = sqrt ? (float) Math.sqrt(clamp) : clamp;
        for (int j = 0; j < hidden; j++) {
            sum[j] /= divisor;
        }
        return new Embedding(sum, new long[] { hidden });
    }

    private Embedding maxPool(float[] values, long[] shape, long[] attentionMask) {
        int hidden = hiddenSize(shape, attentionMask);
        float[] max = new float[hidden];
        for (int i = 0; i < attentionMask.length; i++) {
            int offset = i * hidden;
            for (int j = 0; j < hidden; j++) {
                // padding tokens are set to a large negative value
                float value = attentionMask[i] == 0 ? MAX_POOLING_PADDING_VALUE : values[offset + j];
                max[j] = i == 0 ? value : Math.max(max[j], value);
            }
        }
        // the max is taken with keepDims, so the result keeps a leading dimension of 1
        return new Embedding(max, new long[] { 1, hidden });
    }

    private Embedding weightedMeanPool(float[] values, long[] shape, long[] attentionMask) {
        int hidden = hiddenSize(shape, attentionMask);
        float[] sum = new float[hidden];
        long weightSum = 0;
        for (int i = 0; i < attentionMask.length; i++) {
            long weight = attentionMask[i] * (i + 1);
            weightSum += weight;
            float floatWeight = weight;
            int offset = i * hidden;
            for (int j = 0; j < hidden; j++) {
                sum[j] += values[offset + j] * floatWeight;
            }
        }
        float divisor = weightSum;
        for (int j = 0; j < hidden; j++) {
            sum[j] /= divisor;
        }
        return new Embedding(sum, new long[] { hidden });
    }

    private Embedding row(float[] values, long[] shape, int index) {
        if (shape.length != 2 || index >= shape[0]) {
            throw new IllegalArgumentException(
                "Unsupported ONNX model output shape " + Arrays.toString(shape) + " for pooling mode " + poolingMode
            );
        }
        int hidden = (int) shape[1];
        return new Embedding(Arrays.copyOfRange(values, index * hidden, (index + 1) * hidden), new long[] { hidden });
    }

    /** Index of the last non-padding token; works for left- and right-padded input, 0 if the mask is all padding. */
    private static int findLastRealTokenIndex(long[] attentionMask) {
        for (int i = attentionMask.length - 1; i >= 0; i--) {
            if (attentionMask[i] != 0L) {
                return i;
            }
        }
        return 0;
    }

    /** L2-normalizes along the first dimension: x / max(||x||, 1e-12). */
    private static void normalize(Embedding embedding) {
        float[] values = embedding.values;
        if (values.length == 0) {
            return;
        }
        int rows = (int) embedding.shape[0];
        int stride = values.length / rows;
        for (int k = 0; k < stride; k++) {
            double squareSum = 0;
            for (int i = 0; i < rows; i++) {
                float value = values[i * stride + k];
                squareSum += (double) value * value;
            }
            float divisor = Math.max((float) Math.sqrt(squareSum), NORMALIZE_EPSILON);
            for (int i = 0; i < rows; i++) {
                values[i * stride + k] /= divisor;
            }
        }
    }

    @Override
    public void setArguments(Map<String, ?> arguments) {}
}
