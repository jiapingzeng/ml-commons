/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.stream.IntStream;

import org.opensearch.core.action.ActionListener;
import org.opensearch.ml.action.agenticindex.BoundedFanOut.Outcome;
import org.opensearch.test.OpenSearchTestCase;

public class BoundedFanOutTests extends OpenSearchTestCase {

    public void testBoundsConcurrencyAndKeepsItemOrder() {
        List<ActionListener<Integer>> held = new ArrayList<>();
        AtomicReference<List<Outcome<Integer>>> done = new AtomicReference<>();
        List<Integer> items = List.of(1, 2, 3, 4, 5);

        BoundedFanOut.<Integer, Integer>run(items, 2, (item, listener) -> held.add(listener), done::set);

        assertEquals(2, held.size());
        // Completing out of order starts the next items and keeps outcomes in item order.
        held.get(1).onResponse(20);
        assertEquals(3, held.size());
        held.get(0).onFailure(new IllegalStateException("one failed"));
        held.get(2).onResponse(30);
        held.get(3).onResponse(40);
        assertNull(done.get());
        held.get(4).onResponse(50);
        List<Outcome<Integer>> outcomes = done.get();
        assertEquals("one failed", outcomes.get(0).failure().getMessage());
        assertEquals(List.of(20, 30, 40, 50), outcomes.subList(1, 5).stream().map(Outcome::result).toList());
    }

    public void testSynchronousStepsDoNotGrowTheStack() {
        List<Integer> items = IntStream.range(0, 20000).boxed().toList();
        AtomicReference<List<Outcome<Integer>>> done = new AtomicReference<>();
        AtomicInteger calls = new AtomicInteger();

        BoundedFanOut.<Integer, Integer>run(items, 4, (item, listener) -> {
            calls.incrementAndGet();
            listener.onResponse(item * 2);
        }, done::set);

        assertEquals(20000, calls.get());
        assertEquals(Integer.valueOf(39998), done.get().get(19999).result());
    }

    public void testStepThatThrowsIsAFailure() {
        AtomicReference<List<Outcome<Integer>>> done = new AtomicReference<>();
        BoundedFanOut.<Integer, Integer>run(List.of(1), 1, (item, listener) -> { throw new IllegalArgumentException("bad"); }, done::set);
        assertEquals("bad", done.get().get(0).failure().getMessage());
    }

    public void testEmptyItemsCompleteImmediately() {
        AtomicReference<List<Outcome<Integer>>> done = new AtomicReference<>();
        BoundedFanOut.<Integer, Integer>run(List.of(), 3, (item, listener) -> fail("no items"), done::set);
        assertEquals(List.of(), done.get());
    }

    public void testOnDoneExceptionPropagatesAndCountsOnce() {
        AtomicInteger doneCalls = new AtomicInteger();
        IllegalStateException e = expectThrows(
            IllegalStateException.class,
            () -> BoundedFanOut.<Integer, Integer>run(List.of(1, 2), 2, (item, listener) -> listener.onResponse(item), outcomes -> {
                doneCalls.incrementAndGet();
                throw new IllegalStateException("onDone failed");
            })
        );
        assertEquals("onDone failed", e.getMessage());
        assertEquals(1, doneCalls.get());
    }

    public void testOnDoneExceptionOnAsyncCompletionPropagatesToCompleter() {
        List<ActionListener<Integer>> held = new ArrayList<>();
        AtomicInteger doneCalls = new AtomicInteger();
        BoundedFanOut.<Integer, Integer>run(List.of(1), 1, (item, listener) -> held.add(listener), outcomes -> {
            doneCalls.incrementAndGet();
            throw new IllegalStateException("onDone failed");
        });
        expectThrows(IllegalStateException.class, () -> held.get(0).onResponse(1));
        // A late failure for the completed item is ignored rather than counted again.
        held.get(0).onFailure(new IllegalArgumentException("late"));
        assertEquals(1, doneCalls.get());
    }
}
