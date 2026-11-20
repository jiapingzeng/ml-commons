/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.text_embedding;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.opensearch.ml.common.model.BaseModelConfig.FrameworkType.HUGGINGFACE_TRANSFORMERS;
import static org.opensearch.ml.engine.algorithms.DLModel.ML_ENGINE;
import static org.opensearch.ml.engine.algorithms.DLModel.MODEL_HELPER;
import static org.opensearch.ml.engine.algorithms.DLModel.MODEL_ZIP_FILE;
import static org.opensearch.ml.engine.algorithms.text_embedding.TextEmbeddingDenseModel.SENTENCE_EMBEDDING;

import java.io.File;
import java.util.List;
import java.util.Map;

import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.MLModel;
import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.model.BaseModelConfig.PoolingMode;
import org.opensearch.ml.common.model.MLModelFormat;
import org.opensearch.ml.common.model.MLModelState;
import org.opensearch.ml.common.model.TextEmbeddingModelConfig;
import org.opensearch.ml.common.output.model.ModelTensor;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.common.output.model.ModelTensors;
import org.opensearch.ml.engine.MLEngine;
import org.opensearch.ml.engine.ModelHelper;
import org.opensearch.ml.engine.encryptor.Encryptor;
import org.opensearch.ml.engine.encryptor.EncryptorImpl;

/**
 * Deploys and runs ONNX text embedding models in a JVM started with ai.djl.onnx.disable_alternative=true, where any
 * NDArray operation that needs DJL's PyTorch alternative engine fails. Runs in its own Gradle test task because DJL
 * keeps the alternative engine for the life of the JVM once another test resolves it.
 */
public class ONNXTextEmbeddingWithoutAlternativeEngineTest {
    private static final List<String> DOCS = List.of("today is sunny", "That is a happy dog", "");

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    @Test
    public void predict_bert_allPoolingModes() throws Exception {
        assertPredictsWithoutAlternativeEngine("all-MiniLM-L6-v2_onnx.zip", "bert", 384);
    }

    @Test
    public void predict_albert_allPoolingModes() throws Exception {
        assertPredictsWithoutAlternativeEngine("paraphrase-albert-small-v2_onnx.zip", "albert", 768);
    }

    private void assertPredictsWithoutAlternativeEngine(String zipName, String modelType, int dimension) throws Exception {
        assertEquals(
            "run with ./gradlew :opensearch-ml-algorithms:onnxWithoutAlternativeEngineTest",
            "true",
            System.getProperty("ai.djl.onnx.disable_alternative")
        );
        Encryptor encryptor = new EncryptorImpl(null, "m+dWmfmnNRiNlOdej/QelEkvMTyH//frS2TBeS2BP4w=");
        MLEngine mlEngine = new MLEngine(tmp.newFolder("ml_cache").toPath(), encryptor);
        File zip = new File(getClass().getResource(zipName).toURI());
        for (PoolingMode poolingMode : PoolingMode.values()) {
            for (boolean normalize : new boolean[] { true, false }) {
                String label = modelType + " " + poolingMode + " normalize=" + normalize;
                TextEmbeddingDenseModel model = new TextEmbeddingDenseModel();
                try {
                    model
                        .initModel(
                            mlModel(modelType, poolingMode, normalize, dimension),
                            Map.of(MODEL_ZIP_FILE, zip, MODEL_HELPER, new ModelHelper(mlEngine), ML_ENGINE, mlEngine),
                            encryptor
                        );
                    MLInput input = MLInput
                        .builder()
                        .algorithm(FunctionName.TEXT_EMBEDDING)
                        .inputDataset(TextDocsInputDataSet.builder().docs(DOCS).build())
                        .build();
                    List<ModelTensors> outputs = ((ModelTensorOutput) model.predict(input)).getMlModelOutputs();
                    assertEquals(label, DOCS.size(), outputs.size());
                    for (ModelTensors tensors : outputs) {
                        ModelTensor tensor = tensors.getMlModelTensors().get(0);
                        assertEquals(label, SENTENCE_EMBEDDING, tensor.getName());
                        if (poolingMode == PoolingMode.NONE) {
                            // without a pre-pooled output the token embeddings are returned as-is
                            assertEquals(label, 0, tensor.getData().length % dimension);
                        } else {
                            assertEquals(label, dimension, tensor.getData().length);
                        }
                    }
                } finally {
                    model.close();
                }
            }
        }
        // predict not throwing is the check: with the property true, any op needing the alternative engine throws
        assertTrue(Boolean.getBoolean("ai.djl.onnx.disable_alternative"));
    }

    private static MLModel mlModel(String modelType, PoolingMode poolingMode, boolean normalize, int dimension) {
        return MLModel
            .builder()
            .modelFormat(MLModelFormat.ONNX)
            .name("test_model_name")
            .modelId("test_model_id")
            .algorithm(FunctionName.TEXT_EMBEDDING)
            .version("1.0.0")
            .modelState(MLModelState.TRAINED)
            .modelConfig(
                TextEmbeddingModelConfig
                    .builder()
                    .modelType(modelType)
                    .embeddingDimension(dimension)
                    .frameworkType(HUGGINGFACE_TRANSFORMERS)
                    .poolingMode(poolingMode)
                    .normalizeResult(normalize)
                    .build()
            )
            .build();
    }
}
