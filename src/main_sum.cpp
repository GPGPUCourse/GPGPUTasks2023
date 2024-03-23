#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include "cl/sum_cl.h"
#include "libgpu/shared_device_buffer.h"
#include "libgpu/context.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

const unsigned int WORKGROUP_SIZE = 64;
const unsigned int VALUES_PER_WORKITEM = 32;
void runBenchmark(gpu::gpu_mem_32u& arr,
                gpu::gpu_mem_32u& gpuSum,
                const std::string& kernelName,
                const int benchmarkingIters,
                const uint reference_sum,
                const uint n) {
    ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernelName);
    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        cl_uint sum = 0;
        gpuSum.writeN(&sum, 1);
        kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, n),
                    arr.clmem(),
                    gpuSum.clmem(),
                    n);
        gpuSum.readN(&sum, 1);
        EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
        t.nextLap();
    }
    t.stop();
    std::cout << "GPU " + kernelName + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU " + kernelName + ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u arr;
        gpu::gpu_mem_32u gpuSum;
        arr.resizeN(n);
        gpuSum.resizeN(1);

        arr.writeN(&as[0], n);

        runBenchmark(arr, gpuSum, "atomicSum1", benchmarkingIters, reference_sum, n);
        runBenchmark(arr, gpuSum, "loopSum2", benchmarkingIters, reference_sum, n);
        runBenchmark(arr, gpuSum, "loopCoalescedSum3", benchmarkingIters, reference_sum, n);
        runBenchmark(arr, gpuSum, "localMemSum4", benchmarkingIters, reference_sum, n);
        runBenchmark(arr, gpuSum, "treeSum5", benchmarkingIters, reference_sum, n);
    }
}
