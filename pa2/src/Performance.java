import java.util.Random;
import java.util.concurrent.BrokenBarrierException;
import java.util.concurrent.CyclicBarrier;

public class Performance implements Runnable {

    private final int numThreads;
    private final int keyRange;
    private final int readPercent;
    private final IntSet impl;
    private final int testTime;
    private final boolean verbose;

    private volatile boolean done;

    public Performance(int keyRange,
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
        final long[] results = new long[numThreads];

        for (int i = 0; i < threads.length; ++i) {
            final int index = i;
            threads[i] = new Thread() {
                @Override
				public void run() {
                    final Random rand = new Random(index);
                    try {
                        barrier.await();
                    } catch (final InterruptedException xx) {
                        throw new Error("unexpected", xx);
                    } catch (final BrokenBarrierException xx) {
                        throw new RuntimeException("unexpected", xx);
                    }
                    int count = 0;
                    while (!done) {
                        final int key = rand.nextInt(keyRange);
                        final int pct = rand.nextInt(200);
                        if (pct < readPercent * 2) {
                            impl.contains(key);
                        } else if ((pct & 1) == 0) {
                            impl.add(key);
                        } else {
                            impl.remove(key);
                        }
                        ++count;
                    }
                    results[index] = count;
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

        long total = 0;
        for (long c : results) {
            total += c;
        }

        if (verbose) {
            System.out.printf("%18s, %2d threads: %9.0f operations/sec %9.0f operations/sec/thread\n",
                              impl.getClass().getSimpleName(), numThreads,
                              total * 1000.0 / testTime,
                              total * 1000.0 / testTime / numThreads);
        }
    }
}
