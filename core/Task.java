package core;

import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public abstract class Task {
    // the number of CPU cores this computer has
    private static final int cpuNum = Runtime.getRuntime().availableProcessors();

    // an ExecutorService with the same number of threads as the number of CPU cores
    private static final ExecutorService exec = Executors.newFixedThreadPool(cpuNum);

    // how many tasks can be run independently
    // for conv network, each 2D layer can be processed independently,
    // so workLength is usually the depth of a 3D matrix
    private int workLength;

    /**
     * Class constructor. Take the number of tasks to run.
     * @param workLength the number of tasks.
     */
    public Task( int workLength ) {
        this.workLength = workLength;
    }

    /**
     * Start working on all of the tasks simultaneously by dividing them to all CPU cores.
     */
    public void start() {
        // how many CPU cores do we need?
        int runCpu = cpuNum < workLength ? cpuNum : 1;

        // use a CountDownLatch to hold other threads waiting until all threads started here have finished
        CountDownLatch gate = new CountDownLatch(runCpu);

        // how many tasks does each CPU core handle?
        // this is ceiling( workLength / runCPU )
        int fregLength = (workLength + runCpu - 1) / runCpu;

        // assign tasks to each CPU
        for (int cpuIndex = 0; cpuIndex < runCpu; cpuIndex++) {
            // CPU at cpuIndex takes all the tasks from cpuIndex * fregLength to (cpuIndex+1) * fregLength
            int start = cpuIndex * fregLength;
            int tmp = (cpuIndex + 1) * fregLength;
            int end = tmp <= workLength ? tmp : workLength;
            // fire a thread
            Runnable task = new Runnable() {

                @Override
                public void run() {
                    process(start, end);
                    // signal that one thread has finished
                    gate.countDown();
                }

            };
            exec.execute(task);
        }
        try {
            // wait until all threads started here have finished
            gate.await();
        } catch (InterruptedException e) {
            e.printStackTrace();
            throw new RuntimeException(e);
        }
    }

    /**
     * Process tasks that are assigned to one CPU core.
     * @param start The index of the first task to process.
     * @param end the index of the last task to process.
     */
    public abstract void process(int start, int end);

}