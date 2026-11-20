/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.opensearch.ml.engine.algorithms.DLModel.ONNX_DISABLE_ALTERNATIVE;

import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

import org.junit.After;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.engine.algorithms.question_answering.QuestionAnsweringModel;
import org.opensearch.ml.engine.algorithms.sparse_encoding.TextEmbeddingSparseEncodingModel;
import org.opensearch.ml.engine.algorithms.text_embedding.TextEmbeddingDenseModel;
import org.opensearch.ml.engine.algorithms.text_similarity.TextSimilarityCrossEncoderModel;
import org.opensearch.ml.engine.algorithms.tokenize.SparseTokenizerModel;

import ai.djl.engine.Engine;
import ai.djl.engine.EngineException;

public class DLModelOnnxAlternativeEngineTest {
    private static final String MODEL_ID = "test_model_id";
    private static final String[] SAVED_PROPERTIES = { ONNX_DISABLE_ALTERNATIVE, "DJL_CACHE_DIR", "java.library.path" };

    @Rule
    public TemporaryFolder tmp = new TemporaryFolder();

    private final Map<String, String> savedProperties = new HashMap<>();

    @BeforeClass
    public static void initDLModelWithPropertyCleared() throws ClassNotFoundException {
        String previous = System.getProperty(ONNX_DISABLE_ALTERNATIVE);
        System.clearProperty(ONNX_DISABLE_ALTERNATIVE);
        try {
            // DLModel reads the startup value at class init; neither the constant import nor DLModel.class triggers it
            Class.forName(DLModel.class.getName(), true, DLModel.class.getClassLoader());
        } finally {
            if (previous != null) {
                System.setProperty(ONNX_DISABLE_ALTERNATIVE, previous);
            }
        }
    }

    @Before
    public void setUp() throws IOException {
        for (String key : SAVED_PROPERTIES) {
            savedProperties.put(key, System.getProperty(key));
        }
        String cache = tmp.newFolder("ml_cache").getAbsolutePath();
        // same DJL setup as DLModel.loadModel, used if the PyTorch engine is not loaded yet
        System.setProperty("PYTORCH_PRECXX11", "true");
        System.setProperty("PYTORCH_VERSION", "2.5.1");
        System.setProperty("DJL_CACHE_DIR", cache);
        System.setProperty("java.library.path", cache);
        System.setProperty("ai.djl.pytorch.num_interop_threads", "1");
        System.setProperty("ai.djl.pytorch.num_threads", "1");
    }

    @After
    public void tearDown() {
        savedProperties.forEach((key, value) -> {
            if (value == null) {
                System.clearProperty(key);
            } else {
                System.setProperty(key, value);
            }
        });
    }

    @Test
    public void notRequired_unset_disablesAlternativeEngine() {
        System.clearProperty(ONNX_DISABLE_ALTERNATIVE);
        configure(FunctionName.TEXT_EMBEDDING, false);
        assertEquals("true", System.getProperty(ONNX_DISABLE_ALTERNATIVE));
    }

    @Test
    public void notRequired_afterRequired_keepsAlternativeEngine() {
        System.setProperty(ONNX_DISABLE_ALTERNATIVE, "false");
        configure(FunctionName.TEXT_EMBEDDING, false);
        assertEquals("false", System.getProperty(ONNX_DISABLE_ALTERNATIVE));
    }

    @Test
    public void required_enablesAlternativeEngine() {
        System.setProperty(ONNX_DISABLE_ALTERNATIVE, "true");
        configure(FunctionName.SPARSE_ENCODING, true);
        assertEquals("false", System.getProperty(ONNX_DISABLE_ALTERNATIVE));
        assertNotNull(Engine.getEngine("OnnxRuntime").getAlternativeEngine());

        configure(FunctionName.TEXT_EMBEDDING, false);
        assertEquals("false", System.getProperty(ONNX_DISABLE_ALTERNATIVE));
    }

    @Test
    public void required_unset_enablesAlternativeEngine() {
        System.clearProperty(ONNX_DISABLE_ALTERNATIVE);
        configure(FunctionName.SPARSE_ENCODING, true);
        assertEquals("false", System.getProperty(ONNX_DISABLE_ALTERNATIVE));
    }

    @Test
    public void required_resolutionFails_restoresPreviousValue() {
        EngineException failure = new EngineException("Failed to load PyTorch native library");
        System.setProperty(ONNX_DISABLE_ALTERNATIVE, "true");
        EngineException e = assertThrows(
            EngineException.class,
            () -> DLModel.configureOnnxAlternativeEngine(MODEL_ID, FunctionName.SPARSE_ENCODING, true, false, () -> {
                throw failure;
            })
        );
        assertSame(failure, e);
        assertEquals("true", System.getProperty(ONNX_DISABLE_ALTERNATIVE));
    }

    @Test
    public void required_resolutionFails_unset_clearsValue() {
        UnsatisfiedLinkError failure = new UnsatisfiedLinkError("libtorch.so");
        System.clearProperty(ONNX_DISABLE_ALTERNATIVE);
        UnsatisfiedLinkError e = assertThrows(
            UnsatisfiedLinkError.class,
            () -> DLModel.configureOnnxAlternativeEngine(MODEL_ID, FunctionName.SPARSE_ENCODING, true, false, () -> {
                throw failure;
            })
        );
        assertSame(failure, e);
        assertNull(System.getProperty(ONNX_DISABLE_ALTERNATIVE));
    }

    @Test
    public void required_disabledAtStartup_failsWithoutOverriding() {
        System.setProperty(ONNX_DISABLE_ALTERNATIVE, "true");
        IllegalArgumentException e = assertThrows(
            IllegalArgumentException.class,
            () -> DLModel.configureOnnxAlternativeEngine(MODEL_ID, FunctionName.SPARSE_ENCODING, true, true, () -> {
                throw new AssertionError("must not resolve the engine");
            })
        );
        assertEquals(
            "ONNX SPARSE_ENCODING models need the PyTorch runtime, but this node sets ai.djl.onnx.disable_alternative=true. "
                + "Remove that JVM option, or deploy the model in TORCH_SCRIPT format.",
            e.getMessage()
        );
        assertEquals("true", System.getProperty(ONNX_DISABLE_ALTERNATIVE));
    }

    @Test
    public void notRequired_disabledAtStartup_keepsValue() {
        System.setProperty(ONNX_DISABLE_ALTERNATIVE, "true");
        DLModel.configureOnnxAlternativeEngine(MODEL_ID, FunctionName.TEXT_EMBEDDING, false, true, () -> {
            throw new AssertionError("must not resolve the engine");
        });
        assertEquals("true", System.getProperty(ONNX_DISABLE_ALTERNATIVE));
    }

    @Test
    public void requiresOnnxAlternativeEngine_perModel() {
        assertFalse(requires(new TextEmbeddingDenseModel()));
        assertFalse(requires(new SparseTokenizerModel()));
        assertTrue(requires(new TextEmbeddingSparseEncodingModel()));
        assertTrue(requires(new TextSimilarityCrossEncoderModel()));
        assertTrue(requires(new QuestionAnsweringModel()));
    }

    // disabledAtStartup=false so the result never depends on the property value when DLModel was initialized
    private static void configure(FunctionName functionName, boolean required) {
        DLModel
            .configureOnnxAlternativeEngine(
                MODEL_ID,
                functionName,
                required,
                false,
                () -> Engine.getEngine("OnnxRuntime").getAlternativeEngine()
            );
    }

    private static boolean requires(DLModel model) {
        return model.requiresOnnxAlternativeEngine();
    }
}
