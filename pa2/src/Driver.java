
public class Driver {

    private static final String DEFAULT_TEST = "correctness";
    private static final int DEFAULT_KEY_RANGE = 10000;
    private static final int DEFAULT_READ_PERCENT = 95;
    private static final int DEFAULT_MAX_THREADS = 16;
    private static final int DEFAULT_WARMUP_TIME = 250;
    private static final int DEFAULT_TEST_TIME = 2000;

    private static IntSet newImpl(int i) {
        switch (i) {
        case 0:
            return new CoarseLockTreap();
        case 1:
            return new STMTreap();
        default:
            return null;
        }
    }

    private static Runnable newTestRunner(String testRunnerName,
                                          int keyRange,
                                          int readPercent,
                                          int threads,
                                          int testTime,
                                          IntSet impl,
                                          boolean verbose) {
        if (testRunnerName.equalsIgnoreCase("correctness")) {
            return new Correctness(keyRange, readPercent, threads, testTime, impl, verbose);
        } else if (testRunnerName.equalsIgnoreCase("performance")) {
            return new Performance(keyRange, readPercent, threads, testTime, impl, verbose);
        }

        return null;
    }

    private static void usage() {
        System.out.println(
            "Usage:\n" +
            "  java -javaagent:deuceAgent-1.3.0.jar " + Driver.class.getName() + " [options]\n" +
            "Where options are:\n" +
            "  --test=N          Either correctness or performance (default " + DEFAULT_TEST + ")\n" +
            "  --key-range=N     Number of unique keys to use (default " + DEFAULT_KEY_RANGE + ")\n" +
            "  --read-percent=N  Percentage of ops that are reads (default " + DEFAULT_READ_PERCENT + ")\n" +
            "  --max-threads=N   Powers of 2 up to N max threads (default " + DEFAULT_MAX_THREADS + ")\n" +
            "  --warmup=N        Warm up N ms each (default " + DEFAULT_WARMUP_TIME + ")\n" +
            "  --timed=N         Test for N ms each (default " + DEFAULT_TEST_TIME + ")\n");
        System.exit(-1);
    }

    public static void main(String[] args) {
        String testRunnerName = DEFAULT_TEST;
        int keyRange = DEFAULT_KEY_RANGE;
        int readPercent = DEFAULT_READ_PERCENT;
        int maxThreads = DEFAULT_MAX_THREADS;
        int warmupTime = DEFAULT_WARMUP_TIME;
        int testTime = DEFAULT_TEST_TIME;

        for (String arg : args) {
            int equals = arg.indexOf('=');
            if (equals < 0) {
                usage();
            }

            String name = arg.substring(0, equals);
            String value = arg.substring(equals + 1);
            if (name.equals("--test")) {
                if (value.equalsIgnoreCase("correctness")) {
                    testRunnerName = value;
                } else if (value.equalsIgnoreCase("performance")) {
                    testRunnerName = value;
                } else {
                    usage();
                }
            } else if (name.equals("--key-range")) {
                keyRange = Integer.parseInt(value);
            } else if (name.equals("--read-percent")) {
                readPercent = Integer.parseInt(value);
            } else if (name.equals("--max-threads")) {
                maxThreads = Integer.parseInt(value);
            } else if (name.equals("--warmup")) {
                warmupTime = Integer.parseInt(value);
            } else if (name.equals("--timed")) {
                testTime = Integer.parseInt(value);
            } else {
                usage();
            }
        }

        if (testRunnerName.equalsIgnoreCase("performance")) {
            System.out.printf("Warming up...\n", keyRange, readPercent);
            outer: for (int implNum = 0;; implNum++) {
                for (int threads = 1; threads <= maxThreads; threads *= 2) {
                    IntSet impl = newImpl(implNum);
                    if (impl == null) {
                        break outer;
                    }

                    newTestRunner(testRunnerName, keyRange, readPercent, threads, warmupTime, impl, false).run();
                }
            }
        }

        System.out.printf("Test run for %7d range, %2d%% read...\n", keyRange, readPercent);
        outer: for (int implNum = 0;; implNum++) {
            for (int threads = 1; threads <= maxThreads; threads *= 2) {
                IntSet impl = newImpl(implNum);
                if (impl == null) {
                    break outer;
                }

                newTestRunner(testRunnerName, keyRange, readPercent, threads, testTime, impl, true).run();
            }
        }
    }
}
