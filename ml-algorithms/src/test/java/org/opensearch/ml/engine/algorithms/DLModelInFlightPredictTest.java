/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertSame;
import static org.junit.Assert.assertThrows;
import static org.junit.Assert.assertTrue;
import static org.mockito.Mockito.doThrow;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.never;
import static org.mockito.Mockito.times;
import static org.mockito.Mockito.verify;

import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.opensearch.ml.common.FunctionName;
import org.opensearch.ml.common.dataset.TextDocsInputDataSet;
import org.opensearch.ml.common.exception.MLException;
import org.opensearch.ml.common.input.MLInput;
import org.opensearch.ml.common.model.MLModelConfig;
import org.opensearch.ml.common.output.MLOutput;
import org.opensearch.ml.common.output.model.ModelTensorOutput;
import org.opensearch.ml.engine.ModelHelper;

import ai.djl.Device;
import ai.djl.inference.Predictor;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.Translator;
import ai.djl.translate.TranslatorFactory;

/**
 * Undeploy releases a model while predict threads may still be running on its predictors. The predictors and models
 * must stay open until the last in-flight predict finishes, and predicts that start after release must fail cleanly.
 */
public class DLModelInFlightPredictTest {

    private static final MLInput INPUT = MLInput
        .builder()
        .algorithm(FunctionName.TEXT_EMBEDDING)
        .inputDataset(TextDocsInputDataSet.builder().docs(List.of("doc")).build())
        .build();

    private Predictor<Input, Output> predictor;
    private ZooModel<Input, Output> zooModel;
    private BlockingModel model;
    private ExecutorService executor;

    @Before
    @SuppressWarnings("unchecked")
    public void setUp() {
        predictor = mock(Predictor.class);
        zooModel = mock(ZooModel.class);
        model = new BlockingModel();
        model.modelHelper = mock(ModelHelper.class);
        model.modelId = "model_id";
        model.devices = new Device[] { Device.cpu() };
        model.predictors = new Predictor[] { predictor };
        model.models = new ZooModel[] { zooModel };
        executor = Executors.newFixedThreadPool(3);
    }

    @After
    public void tearDown() {
        model.finish.countDown();
        executor.shutdownNow();
    }

    @Test
    public void release_WithPredictsInFlight_ClosesAfterTheLastOneFinishes() throws Exception {
        model.expectEntries(2);
        Future<MLOutput> first = executor.submit(() -> model.predict(INPUT));
        Future<MLOutput> second = executor.submit(() -> model.predict(INPUT));
        assertTrue(model.entered.await(10, TimeUnit.SECONDS));

        // release must not wait for the in-flight predicts
        executor.submit(model::release).get(10, TimeUnit.SECONDS);
        verify(predictor, never()).close();
        verify(zooModel, never()).close();

        // a predict that starts after release fails cleanly and never reaches the predictors
        MLException e = assertThrows(MLException.class, () -> model.predict(INPUT));
        assertEquals("model not deployed.", e.getCause().getMessage());
        assertEquals(2, model.calls.get());

        model.finish.countDown();
        assertNotNull(first.get(10, TimeUnit.SECONDS));
        assertNotNull(second.get(10, TimeUnit.SECONDS));
        verify(predictor, times(1)).close();
        verify(zooModel, times(1)).close();

        // a later undeploy finds nothing left to close
        model.close();
        model.release();
        verify(predictor, times(1)).close();
        verify(zooModel, times(1)).close();
    }

    @Test
    public void release_WithFailingPredictInFlight_StillCloses() throws Exception {
        model.failure = new IllegalStateException("inference failed");
        model.expectEntries(1);
        Future<MLOutput> inFlight = executor.submit(() -> model.predict(INPUT));
        assertTrue(model.entered.await(10, TimeUnit.SECONDS));

        model.release();
        verify(predictor, never()).close();

        model.finish.countDown();
        Exception e = assertThrows(Exception.class, () -> inFlight.get(10, TimeUnit.SECONDS));
        assertSame(model.failure, e.getCause().getCause());
        verify(predictor, times(1)).close();
        verify(zooModel, times(1)).close();
    }

    @Test
    public void release_CloseFailureAfterLastPredict_DoesNotFailThatPredict() throws Exception {
        doThrow(new IllegalStateException("close failed")).when(predictor).close();
        model.expectEntries(1);
        Future<MLOutput> inFlight = executor.submit(() -> model.predict(INPUT));
        assertTrue(model.entered.await(10, TimeUnit.SECONDS));

        model.release();
        model.finish.countDown();

        assertNotNull(inFlight.get(10, TimeUnit.SECONDS));
        verify(predictor, times(1)).close();
    }

    @Test
    public void release_WithNoPredictInFlight_ClosesImmediately() {
        model.finish.countDown();
        assertNotNull(model.predict(INPUT));

        model.release();

        verify(predictor, times(1)).close();
        verify(zooModel, times(1)).close();
        assertFalse(model.isModelReady());
        MLException e = assertThrows(MLException.class, () -> model.predict(INPUT));
        assertEquals("model not deployed.", e.getCause().getMessage());
    }

    private static class BlockingModel extends DLModel {
        private final CountDownLatch finish = new CountDownLatch(1);
        private final AtomicInteger calls = new AtomicInteger();
        private CountDownLatch entered = new CountDownLatch(0);
        private RuntimeException failure;

        void expectEntries(int count) {
            entered = new CountDownLatch(count);
        }

        @Override
        public ModelTensorOutput predict(String modelId, MLInput input) {
            calls.incrementAndGet();
            entered.countDown();
            try {
                if (!finish.await(10, TimeUnit.SECONDS)) {
                    throw new IllegalStateException("predict was not released");
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
            if (failure != null) {
                throw failure;
            }
            // the predictors are still open while this call runs
            getPredictor();
            return new ModelTensorOutput(List.of());
        }

        @Override
        public Translator<Input, Output> getTranslator(String engine, MLModelConfig modelConfig) {
            return null;
        }

        @Override
        public TranslatorFactory getTranslatorFactory(String engine, MLModelConfig modelConfig) {
            return null;
        }
    }
}
