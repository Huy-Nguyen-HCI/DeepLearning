import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;


public class ConcurrentRunner {

    private static final ExecutorService exec;
    public static final int cpuNum;
    static {
        cpuNum = Runtime.getRuntime().availableProcessors();
        exec = Executors.newFixedThreadPool(cpuNum);
    }

    public static void run(Runnable task) {
        exec.execute(task);
    }

    public static void stop() {
        exec.shutdown();
    }

}
