/*
 * Copyright OpenSearch Contributors
 * SPDX-License-Identifier: Apache-2.0
 */

package org.opensearch.ml.action.agenticindex;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReferenceArray;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

import org.opensearch.core.action.ActionListener;

/**
 * Runs one asynchronous step per item with at most {@code concurrency} steps in flight, then reports every item's
 * outcome in item order. Steps that complete on the calling thread are run in a loop rather than recursively, so long
 * item lists do not grow the stack.
 */
final class BoundedFanOut<T, R> {

    /** Outcome of one item: exactly one of result and failure is set. */
    record Outcome<R>(R result, Exception failure) {
    }

    private final List<T> items;
    private final int concurrency;
    private final BiConsumer<T, ActionListener<R>> step;
    private final Consumer<List<Outcome<R>>> onDone;
    private final AtomicReferenceArray<Outcome<R>> outcomes;
    private final AtomicInteger next = new AtomicInteger();
    private final AtomicInteger remaining;

    private BoundedFanOut(List<T> items, int concurrency, BiConsumer<T, ActionListener<R>> step, Consumer<List<Outcome<R>>> onDone) {
        this.items = items;
        this.concurrency = concurrency;
        this.step = step;
        this.onDone = onDone;
        this.outcomes = new AtomicReferenceArray<>(items.size());
        this.remaining = new AtomicInteger(items.size());
    }

    static <T, R> void run(List<T> items, int concurrency, BiConsumer<T, ActionListener<R>> step, Consumer<List<Outcome<R>>> onDone) {
        if (items.isEmpty()) {
            onDone.accept(List.of());
            return;
        }
        BoundedFanOut<T, R> fanOut = new BoundedFanOut<>(items, concurrency, step, onDone);
        for (int i = 0; i < Math.min(concurrency, items.size()); i++) {
            fanOut.pump();
        }
    }

    private void pump() {
        int i;
        while ((i = next.getAndIncrement()) < items.size()) {
            ItemListener listener = new ItemListener(i);
            try {
                step.accept(items.get(i), listener);
            } catch (Exception e) {
                // An exception after the item completed comes from a later stage (such as onDone), not from the step.
                if (!listener.complete(new Outcome<>(null, e))) {
                    throw e;
                }
            }
            if (listener.state.compareAndSet(0, 2)) {
                return;
            }
        }
    }

    /** Completes one item at most once; exceptions from later stages propagate to the caller. */
    private final class ItemListener implements ActionListener<R> {
        private final int idx;
        private final AtomicBoolean done = new AtomicBoolean();
        // 0: step running; 1: completed on this thread; 2: this loop returned, so the completion continues pumping.
        private final AtomicInteger state = new AtomicInteger();

        ItemListener(int idx) {
            this.idx = idx;
        }

        @Override
        public void onResponse(R result) {
            complete(new Outcome<>(result, null));
        }

        @Override
        public void onFailure(Exception e) {
            complete(new Outcome<>(null, e));
        }

        // Returns false when the item had already completed.
        boolean complete(Outcome<R> outcome) {
            if (!done.compareAndSet(false, true)) {
                return false;
            }
            outcomes.set(idx, outcome);
            if (remaining.decrementAndGet() == 0) {
                List<Outcome<R>> all = new ArrayList<>(items.size());
                for (int i = 0; i < items.size(); i++) {
                    all.add(outcomes.get(i));
                }
                onDone.accept(all);
            } else if (!state.compareAndSet(0, 1)) {
                pump();
            }
            return true;
        }
    }
}
