/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.text_embedding;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.io.File;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.stream.Stream;

import org.junit.AfterClass;
import org.junit.BeforeClass;
import org.junit.ClassRule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.opensearch.ml.common.model.BaseModelConfig.PoolingMode;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.engine.utils.ZipUtils;

import ai.djl.Application;
import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.ServingTranslator;
import ai.djl.translate.TranslatorContext;

/**
 * Compares ONNXSentenceTransformerTextEmbeddingTranslator with the NDArray-based translator it replaced
 * (ReferenceONNXTextEmbeddingTranslator), which needs DJL's PyTorch alternative engine.
 */
public class ONNXSentenceTransformerTextEmbeddingTranslatorTest {
    private static final String ONNX_DISABLE_ALTERNATIVE = "ai.djl.onnx.disable_alternative";
    private static final float TOLERANCE = 1e-5f;
    // the reference computes in float16 while the translator converts to float32 first
    private static final float FLOAT16_TOLERANCE = 1e-2f;
    private static final String[] SAVED_PROPERTIES = { ONNX_DISABLE_ALTERNATIVE, "DJL_CACHE_DIR", "java.library.path" };
    private static final List<String> SENTENCES = List
        .of(
            "today is sunny",
            "That is a happy dog",
            "",
            "a",
            "Crème brûlée, naïve café owners and résumé writers",
            "東京は日本の首都です",
            "OpenSearch is a community-driven, open source search and analytics suite. ".repeat(40)
        );
    private static final List<PoolingMode> POOLING_MODES = Stream
        .concat(Arrays.stream(PoolingMode.values()), Stream.of((PoolingMode) null))
        .toList();

    @ClassRule
    public static TemporaryFolder tmp = new TemporaryFolder();

    private static final Map<String, String> savedProperties = new HashMap<>();
    private static ZooModel<Input, Output> bertModel;
    private static ZooModel<Input, Output> albertModel;
    private static NDManager manager;

    @BeforeClass
    public static void setUpClass() throws Exception {
        for (String key : SAVED_PROPERTIES) {
            savedProperties.put(key, System.getProperty(key));
        }
        Path cache = tmp.newFolder("ml_cache").toPath();
        // same DJL setup as DLModel.loadModel
        System.setProperty("PYTORCH_PRECXX11", "true");
        System.setProperty("PYTORCH_VERSION", "2.5.1");
        System.setProperty("DJL_CACHE_DIR", cache.toAbsolutePath().toString());
        System.setProperty("java.library.path", cache.toAbsolutePath().toString());
        System.setProperty("ai.djl.pytorch.num_interop_threads", "1");
        System.setProperty("ai.djl.pytorch.num_threads", "1");
        // the reference translator needs the PyTorch alternative engine
        System.setProperty(ONNX_DISABLE_ALTERNATIVE, "false");
        assertTrue(Engine.getEngine("OnnxRuntime").getAlternativeEngine() != null);

        bertModel = loadModel("all-MiniLM-L6-v2_onnx.zip", "bert");
        albertModel = loadModel("paraphrase-albert-small-v2_onnx.zip", "albert");
        manager = NDManager.newBaseManager("OnnxRuntime");
    }

    @AfterClass
    public static void tearDownClass() {
        if (manager != null) {
            manager.close();
        }
        Stream.of(bertModel, albertModel).filter(m -> m != null).forEach(ZooModel::close);
        savedProperties.forEach((key, value) -> {
            if (value == null) {
                System.clearProperty(key);
            } else {
                System.setProperty(key, value);
            }
        });
    }

    private static ZooModel<Input, Output> loadModel(String zipName, String modelType) throws Exception {
        Path dir = tmp.newFolder(modelType).toPath();
        ZipUtils.unzip(new File(ONNXSentenceTransformerTextEmbeddingTranslatorTest.class.getResource(zipName).toURI()), dir);
        // DLModel names the model file after its directory
        try (Stream<Path> files = Files.list(dir)) {
            Path onnx = files.filter(f -> f.toString().endsWith(".onnx")).findFirst().orElseThrow();
            Files.move(onnx, dir.resolve(dir.getFileName() + ".onnx"));
        }
        return Criteria
            .builder()
            .setTypes(Input.class, Output.class)
            .optApplication(Application.UNDEFINED)
            .optEngine("OnnxRuntime")
            .optModelPath(dir)
            .optTranslator(new ONNXSentenceTransformerTextEmbeddingTranslator(PoolingMode.MEAN, true, modelType))
            .build()
            .loadModel();
    }

    @Test
    public void predict_bert_matchesReference_allPoolingModes() throws Exception {
        assertModelMatchesReference(bertModel, "bert");
    }

    @Test
    public void predict_albert_matchesReference_allPoolingModes() throws Exception {
        assertModelMatchesReference(albertModel, "albert");
    }

    private void assertModelMatchesReference(ZooModel<Input, Output> model, String modelType) throws Exception {
        for (PoolingMode poolingMode : POOLING_MODES) {
            for (boolean normalize : new boolean[] { true, false }) {
                try (
                    Predictor<Input, Output> lean = model
                        .newPredictor(new ONNXSentenceTransformerTextEmbeddingTranslator(poolingMode, normalize, modelType));
                    Predictor<Input, Output> reference = model
                        .newPredictor(new ReferenceONNXTextEmbeddingTranslator(poolingMode, normalize, modelType))
                ) {
                    for (String sentence : SENTENCES) {
                        Input input = new Input();
                        input.add(sentence);
                        String label = modelType + " " + poolingMode + " normalize=" + normalize + " \"" + sentence + "\"";
                        assertSameOutput(label, reference.predict(input), lean.predict(input));
                    }
                }
            }
        }
    }

    @Test
    public void processInput_matchesReference_withAndWithoutTokenTypeIds() throws Exception {
        for (String modelType : new String[] { "bert", "albert", "ALBERT", "roberta", "distilbert", null }) {
            ServingTranslator lean = new ONNXSentenceTransformerTextEmbeddingTranslator(PoolingMode.MEAN, true, modelType);
            ServingTranslator reference = new ReferenceONNXTextEmbeddingTranslator(PoolingMode.MEAN, true, modelType);
            try (NDManager predictorManager = manager.newSubManager(); NDManager ctxManager = manager.newSubManager()) {
                TranslatorContext ctx = context(bertModel, ctxManager, predictorManager, null);
                lean.prepare(ctx);
                reference.prepare(ctx);
                for (String sentence : SENTENCES) {
                    Input input = new Input();
                    input.add(sentence);
                    NDList expected = reference.processInput(ctx, input);
                    NDList actual = lean.processInput(ctx, input);
                    int expectedSize = "bert".equalsIgnoreCase(modelType) || "albert".equalsIgnoreCase(modelType) ? 3 : 2;
                    assertEquals(expectedSize, actual.size());
                    assertEquals(expected.size(), actual.size());
                    for (int i = 0; i < expected.size(); i++) {
                        assertEquals(expected.get(i).getName(), actual.get(i).getName());
                        assertEquals(expected.get(i).getShape(), actual.get(i).getShape());
                        assertEquals(expected.get(i).getDataType(), actual.get(i).getDataType());
                        assertArrayEquals(expected.get(i).toLongArray(), actual.get(i).toLongArray());
                    }
                }
            }
        }
    }

    @Test
    public void processOutput_matchesReference_paddedAndUnpaddedMasks() throws Exception {
        for (DataType dataType : new DataType[] { DataType.FLOAT32, DataType.FLOAT64, DataType.FLOAT16 }) {
            assertProcessOutputMatchesReference(dataType);
        }
    }

    private void assertProcessOutputMatchesReference(DataType dataType) {
        Random random = new Random(42);
        long[][] masks = { { 1, 1, 1, 1, 1 }, { 1, 1, 1, 0, 0 }, { 0, 0, 1, 1, 1 }, { 1, 0, 1, 0, 1 }, { 1 } };
        int hidden = 8;
        float tolerance = dataType == DataType.FLOAT16 ? FLOAT16_TOLERANCE : TOLERANCE;
        for (long[] mask : masks) {
            float[] tokens = randomValues(random, mask.length * hidden, dataType);
            float[] sentence = randomValues(random, hidden, dataType);
            for (PoolingMode poolingMode : POOLING_MODES) {
                for (boolean normalize : new boolean[] { true, false }) {
                    for (boolean withSentenceEmbedding : new boolean[] { true, false }) {
                        String label = dataType
                            + " "
                            + poolingMode
                            + " normalize="
                            + normalize
                            + " mask="
                            + Arrays.toString(mask)
                            + " sentence_embedding="
                            + withSentenceEmbedding;
                        ServingTranslator reference = new ReferenceONNXTextEmbeddingTranslator(poolingMode, normalize, "bert");
                        float[] sentenceEmbedding = withSentenceEmbedding ? sentence : null;
                        Output expected;
                        if (dataType == DataType.FLOAT16 && poolingMode == PoolingMode.MAX) {
                            // the reference set padding tokens to -1e9, which fails to convert to float16 even with no padding
                            AssertionError e = assertThrows(
                                label,
                                AssertionError.class,
                                () -> processOutput(reference, mask, tokens, hidden, sentenceEmbedding, dataType)
                            );
                            assertTrue(label, e.getCause() instanceof EngineException);
                            expected = processOutput(reference, mask, tokens, hidden, sentenceEmbedding, DataType.FLOAT32);
                        } else {
                            expected = processOutput(reference, mask, tokens, hidden, sentenceEmbedding, dataType);
                        }
                        Output actual = processOutput(
                            new ONNXSentenceTransformerTextEmbeddingTranslator(poolingMode, normalize, "bert"),
                            mask,
                            tokens,
                            hidden,
                            sentenceEmbedding,
                            dataType
                        );
                        assertSameOutput(label, expected, actual, tolerance);
                    }
                }
            }
        }
    }

    @Test
    public void processOutput_unpooledTokenEmbeddingsWithoutBatchDimension_matchesReference() throws Exception {
        long[] mask = { 1, 1, 0 };
        float[] tokens = randomValues(new Random(7), 3 * 4);
        for (PoolingMode poolingMode : POOLING_MODES) {
            for (boolean normalize : new boolean[] { true, false }) {
                String label = poolingMode + " normalize=" + normalize + " rank-2 output";
                ServingTranslator reference = new ReferenceONNXTextEmbeddingTranslator(poolingMode, normalize, "bert");
                Output expected;
                if (poolingMode == PoolingMode.WEIGHTED_MEAN) {
                    // the reference called arange on the ONNX NDManager, which fails for a rank-2 output
                    assertThrows(
                        UnsupportedOperationException.class,
                        () -> reference.processOutput(context(bertModel, manager, manager, encoding(mask)), rank2(tokens))
                    );
                    expected = processOutput(reference, mask, tokens, 4, null, DataType.FLOAT32);
                } else {
                    expected = processOutput(reference, mask, rank2(tokens));
                }
                Output actual = processOutput(
                    new ONNXSentenceTransformerTextEmbeddingTranslator(poolingMode, normalize, "bert"),
                    mask,
                    rank2(tokens)
                );
                assertSameOutput(label, expected, actual);
            }
        }
    }

    private static NDList rank2(float[] tokens) {
        return new NDList(manager.create(tokens, new Shape(3, 4)));
    }

    @Test
    public void processOutput_unexpectedShape_throws() {
        ServingTranslator translator = new ONNXSentenceTransformerTextEmbeddingTranslator(PoolingMode.MEAN, true, "bert");
        NDList list = new NDList(manager.create(new float[8], new Shape(1, 2, 4)));
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> translator.processOutput(context(bertModel, manager, manager, encoding(new long[] { 1, 1, 1 })), list)
        );
        assertEquals(
            "Unsupported ONNX model output shape [2, 4] for pooling mode MEAN, expected [sequence_length, hidden_size] with sequence_length 3",
            e.getMessage()
        );
    }

    @Test
    public void processOutput_unsupportedDataType_throws() {
        ServingTranslator translator = new ONNXSentenceTransformerTextEmbeddingTranslator(PoolingMode.MEAN, true, "bert");
        NDList list = new NDList(manager.create(new int[8], new Shape(1, 2, 4)));
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> translator.processOutput(context(bertModel, manager, manager, encoding(new long[] { 1, 1 })), list)
        );
        assertEquals("Unsupported ONNX model output data type int32, expected float16, float32 or float64", e.getMessage());
    }

    @Test
    public void prepare_tokenizerClosedWithPredictorManager() throws Exception {
        ServingTranslator translator = new ONNXSentenceTransformerTextEmbeddingTranslator(PoolingMode.MEAN, true, "bert");
        NDManager predictorManager = manager.newSubManager();
        TranslatorContext ctx = context(bertModel, manager, predictorManager, null);
        translator.prepare(ctx);
        Input input = new Input();
        input.add("today is sunny");
        translator.processInput(ctx, input);

        predictorManager.close();
        assertThrows(IllegalStateException.class, () -> translator.processInput(ctx, input));
    }

    private static float[] randomValues(Random random, int size) {
        return randomValues(random, size, DataType.FLOAT32);
    }

    /** Random values in [-1, 1), exactly representable in the given data type. */
    private static float[] randomValues(Random random, int size, DataType dataType) {
        float[] values = new float[size];
        for (int i = 0; i < size; i++) {
            values[i] = random.nextFloat() * 2 - 1;
            if (dataType == DataType.FLOAT16) {
                values[i] = Float.float16ToFloat(Float.floatToFloat16(values[i]));
            }
        }
        return values;
    }

    private static Output processOutput(
        ServingTranslator translator,
        long[] mask,
        float[] tokens,
        int hidden,
        float[] sentence,
        DataType dataType
    ) {
        NDList list = new NDList(create(tokens, new Shape(1, mask.length, hidden), dataType));
        if (sentence != null) {
            list.add(create(sentence, new Shape(1, hidden), dataType));
        }
        return processOutput(translator, mask, list);
    }

    private static NDArray create(float[] values, Shape shape, DataType dataType) {
        switch (dataType) {
            case FLOAT32:
                return manager.create(values, shape);
            case FLOAT64:
                double[] doubles = new double[values.length];
                for (int i = 0; i < values.length; i++) {
                    doubles[i] = values[i];
                }
                return manager.create(doubles, shape);
            case FLOAT16:
                ByteBuffer buffer = ByteBuffer.allocateDirect(values.length * 2).order(ByteOrder.nativeOrder());
                for (float value : values) {
                    buffer.putShort(Float.floatToFloat16(value));
                }
                buffer.rewind();
                return manager.create(buffer, shape, DataType.FLOAT16);
            default:
                throw new IllegalArgumentException(dataType.toString());
        }
    }

    private static Output processOutput(ServingTranslator translator, long[] mask, NDList list) {
        try (NDManager ctxManager = manager.newSubManager()) {
            return translator.processOutput(context(bertModel, ctxManager, ctxManager, encoding(mask)), list);
        } catch (Exception e) {
            throw new AssertionError(e);
        }
    }

    private static Encoding encoding(long[] mask) {
        Encoding encoding = mock(Encoding.class);
        when(encoding.getAttentionMask()).thenReturn(mask);
        return encoding;
    }

    private static TranslatorContext context(
        ZooModel<Input, Output> model,
        NDManager ctxManager,
        NDManager predictorManager,
        Encoding encoding
    ) {
        TranslatorContext ctx = mock(TranslatorContext.class);
        when(ctx.getModel()).thenReturn(model);
        when(ctx.getNDManager()).thenReturn(ctxManager);
        when(ctx.getPredictorManager()).thenReturn(predictorManager);
        when(ctx.getAttachment("encoding")).thenReturn(encoding);
        return ctx;
    }

    private static void assertSameOutput(String label, Output expectedOutput, Output actualOutput) {
        assertSameOutput(label, expectedOutput, actualOutput, TOLERANCE);
    }

    /** Asserts both outputs hold the same tensor, within a tolerance relative to max(1, |expected|). */
    private static void assertSameOutput(String label, Output expectedOutput, Output actualOutput, float tolerance) {
        List<ModelTensor> expectedTensors = ModelTensors.fromBytes(expectedOutput.getData().getAsBytes()).getMlModelTensors();
        List<ModelTensor> actualTensors = ModelTensors.fromBytes(actualOutput.getData().getAsBytes()).getMlModelTensors();
        assertEquals(label, 1, expectedTensors.size());
        assertEquals(label, 1, actualTensors.size());
        ModelTensor expected = expectedTensors.get(0);
        ModelTensor actual = actualTensors.get(0);
        assertEquals(label, expected.getName(), actual.getName());
        assertArrayEquals(label, expected.getShape(), actual.getShape());
        assertEquals(label, expected.getDataType(), actual.getDataType());
        assertNull(label, actual.getByteBuffer());
        assertNull(label, actual.getResult());
        assertNull(label, actual.getDataAsMap());
        Number[] expectedData = expected.getData();
        Number[] actualData = actual.getData();
        assertEquals(label, expectedData.length, actualData.length);
        List<String> mismatches = new ArrayList<>();
        for (int i = 0; i < expectedData.length; i++) {
            float e = expectedData[i].floatValue();
            float a = actualData[i].floatValue();
            if (Float.isNaN(e) && Float.isNaN(a) || e == a) {
                continue;
            }
            float diff = Math.abs(e - a);
            // float32 sums in a different order differ in the last bits
            if (!(diff <= tolerance * Math.max(1f, Math.abs(e)))) {
                mismatches.add(i + ": expected " + e + " actual " + a);
            }
        }
        assertTrue(label + " " + mismatches, mismatches.isEmpty());
    }
}
