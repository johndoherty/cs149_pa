import java.util.HashSet;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class Correctness implements Runnable {

    private final int keyRange;
    private final int readPercent;
    private final int numThreads;
    private final int testTime;
    private final IntSet impl;
    private final boolean verbose;

    private volatile boolean done;

    public Correctness(int keyRange,
                       int readPercent,
                       int numThreads,
                       int testTime,
                       IntSet impl,
                       boolean verbose) {
        this.keyRange = keyRange;
        this.readPercent = readPercent;
        this.numThreads = numThreads;
        this.testTime = testTime;
        this.impl = impl;
        this.verbose = verbose;
    }

    @Override
	public void run() {
        try {
            test();
        } catch (final InterruptedException xx) {
            throw new Error("unexpected", xx);
        } catch (final BrokenBarrierException xx) {
            throw new RuntimeException("unexpected", xx);
        }
    }

    private void test() throws InterruptedException, BrokenBarrierException {
        final CyclicBarrier barrier = new CyclicBarrier(numThreads + 1);

        final Thread[] threads = new Thread[numThreads];
        final long[] passed = new long[numThreads];
        final long[] failed = new long[numThreads];

        for (int i = 0; i < threads.length; ++i) {
            final int index = i;
            threads[i] = new Thread() {
                @Override
				public void run() {
                    final Set<Integer> reference = new HashSet<Integer>();
                    final Random rand = new Random(index);
                    try {
                        barrier.await();
                    } catch (final InterruptedException xx) {
                        throw new Error("unexpected", xx);
                    } catch (final BrokenBarrierException xx) {
                        throw new RuntimeException("unexpected", xx);
                    }
                    int passes = 0, failures = 0;
                    while (!done) {
                        final int key = rand.nextInt(keyRange/numThreads)*numThreads + index;
                        final int percent = rand.nextInt(200);
                        if (percent < readPercent * 2) {
                            if (impl.contains(key) == reference.contains(key)) {
                                passes++;
                            } else {
                                failures++;
                            }
                        } else if ((percent % 2) == 0) {
                            impl.add(key);
                            reference.add(key);
                        } else {
                            impl.remove(key);
                            reference.remove(key);
                        }
                    }
                    passed[index] = passes;
                    failed[index] = failures;
                }
            };
        }

        for (Thread t : threads) {
            t.start();
        }
        barrier.await();
        Thread.sleep(testTime);
        done = true;

        for (Thread t : threads) {
            t.join();
        }

        long total_passed = 0, total_failed = 0;
        for (long passes : passed) {
            total_passed += passes;
        }
        for (long failures : failed) {
            total_failed += failures;
        }
        long total = total_passed + total_failed;

        if (verbose) {
            System.out.printf("%18s, %2d threads: %9d failed of %9d total\n",
                              impl.getClass().getSimpleName(),
                              numThreads, total_failed, total);
        }
    }
}
