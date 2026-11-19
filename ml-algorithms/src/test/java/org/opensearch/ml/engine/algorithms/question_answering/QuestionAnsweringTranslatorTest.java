/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.engine.algorithms.question_answering;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import static org.mockito.ArgumentMatchers.any;
import static org.mockito.ArgumentMatchers.anyString;
import static org.mockito.Mockito.doAnswer;
import static org.mockito.Mockito.mock;
import static org.mockito.Mockito.when;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.opensearch.ml.common.output.model.ModelTensors;

import ai.djl.Model;
import ai.djl.huggingface.tokenizers.Encoding;
import ai.djl.huggingface.tokenizers.HuggingFaceTokenizer;
import ai.djl.modality.Input;
import ai.djl.modality.Output;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslatorContext;

public class QuestionAnsweringTranslatorTest {

    private static final String QUESTION_A = "which animal sat on the mat";
    private static final String CONTEXT_A = "the small cat sat on the mat near the dog";
    private static final String QUESTION_B = "which cities are capitals";
    private static final String CONTEXT_B = "paris london tokyo";
    private static final int START = 5;
    private static final int END = 7;

    private Path modelPath;
    private HuggingFaceTokenizer referenceTokenizer;
    private ExecutorService executor;

    @Before
    public void setUp() throws Exception {
        modelPath = Paths.get(getClass().getResource("../tokenize/tokenizer.json").toURI()).getParent();
        referenceTokenizer = HuggingFaceTokenizer.builder().optPadding(true).optTokenizerPath(modelPath.resolve("tokenizer.json")).build();
        executor = Executors.newFixedThreadPool(2);
    }

    @After
    public void tearDown() {
        executor.shutdownNow();
        referenceTokenizer.close();
    }

    @Test
    public void processOutput_ConcurrentRequests_EachUsesItsOwnTokens() throws Exception {
        QuestionAnsweringTranslator translator = new QuestionAnsweringTranslator();
        TranslatorContext prepareCtx = mock(TranslatorContext.class);
        Model model = mock(Model.class);
        when(prepareCtx.getModel()).thenReturn(model);
        when(model.getModelPath()).thenReturn(modelPath);
        translator.prepare(prepareCtx);

        CountDownLatch aEncoded = new CountDownLatch(1);
        CountDownLatch bEncoded = new CountDownLatch(1);

        // A encodes, then B encodes, then A decodes: A must still decode against its own tokens.
        Future<String> answerA = executor.submit(() -> {
            TranslatorContext ctx = requestContext();
            translator.processInput(ctx, input(QUESTION_A, CONTEXT_A));
            aEncoded.countDown();
            assertTrue(bEncoded.await(10, TimeUnit.SECONDS));
            return answer(translator.processOutput(ctx, logits(START, END)));
        });
        Future<String> answerB = executor.submit(() -> {
            assertTrue(aEncoded.await(10, TimeUnit.SECONDS));
            TranslatorContext ctx = requestContext();
            translator.processInput(ctx, input(QUESTION_B, CONTEXT_B));
            bEncoded.countDown();
            return answer(translator.processOutput(ctx, logits(START, END)));
        });

        assertEquals(expectedAnswer(QUESTION_A, CONTEXT_A), answerA.get(30, TimeUnit.SECONDS));
        assertEquals(expectedAnswer(QUESTION_B, CONTEXT_B), answerB.get(30, TimeUnit.SECONDS));
    }

    private String expectedAnswer(String question, String context) {
        Encoding encoding = referenceTokenizer.encode(question, context);
        List<String> tokens = Arrays.asList(encoding.getTokens());
        return referenceTokenizer.buildSentence(tokens.subList(START, END + 1));
    }

    private static TranslatorContext requestContext() {
        TranslatorContext ctx = mock(TranslatorContext.class);
        Map<String, Object> attachments = new ConcurrentHashMap<>();
        doAnswer(invocation -> attachments.put(invocation.getArgument(0), invocation.getArgument(1)))
            .when(ctx)
            .setAttachment(anyString(), any());
        when(ctx.getAttachment(anyString())).thenAnswer(invocation -> attachments.get(invocation.<String>getArgument(0)));
        NDManager manager = mock(NDManager.class);
        when(manager.create(any(long[].class))).thenAnswer(invocation -> mock(NDArray.class));
        when(ctx.getNDManager()).thenReturn(manager);
        return ctx;
    }

    private static Input input(String question, String context) {
        Input input = new Input();
        input.add(question);
        input.add(context);
        return input;
    }

    private static NDList logits(int start, int end) {
        return new NDList(argMaxOf(start), argMaxOf(end));
    }

    private static NDArray argMaxOf(int index) {
        NDArray logits = mock(NDArray.class);
        NDArray argMax = mock(NDArray.class);
        when(logits.argMax()).thenReturn(argMax);
        when(argMax.getLong()).thenReturn((long) index);
        return logits;
    }

    private static String answer(Output output) {
        return (String) ModelTensors.fromBytes(output.getData().getAsBytes()).getMlModelTensors().get(0).getResult();
    }
}
