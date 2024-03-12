#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <numeric>

#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void execute_sum(const std::string& kernelName, const gpu::WorkSize& workSize, const std::vector<unsigned int>& sourceVector, int iter = 10) {
    gpu::gpu_mem_32u gpuSource;
    gpuSource.resizeN(sourceVector.size());
    gpuSource.writeN(sourceVector.data(), sourceVector.size());
    ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernelName);

    bool enableLog = false;
    kernel.compile(enableLog);
    unsigned int expectedSum = std::accumulate(sourceVector.begin(), sourceVector.end(), 0);

    timer t;
    for (int i = 0; i < iter; ++i) {
        t.start();
        unsigned int computedSum = 0;

        uint32_t initialValue = 0;
        gpu::gpu_mem_32u gpuResult;
        gpuResult.resizeN(1);
        gpuResult.writeN(&initialValue, 1);
        kernel.exec(workSize, gpuSource, gpuResult, (unsigned int)sourceVector.size());
        gpuResult.readN(&computedSum, 1);

        EXPECT_THE_SAME(expectedSum, computedSum, "GPU result should be consistent!");
        t.nextLap();
        t.stop();
    }

    std::string outputPrefix = "GPU " + kernelName + ": ";
    std::cout << outputPrefix << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << outputPrefix << (sourceVector.size()/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    const int benchmarkingIter = 10;

    unsigned int reference_sum = 0;
    const unsigned int n = 100*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIter; ++iter) {
            t.start();
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
            t.stop();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIter; ++iter) {
            t.start();
            unsigned int sum = 0;
#pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
            t.stop();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        // gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        unsigned int workGroupSize = 256;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        gpu::WorkSize ws(workGroupSize, global_work_size);

        execute_sum("compute_sum_baseline", ws, as, benchmarkingIter);
        execute_sum("compute_sum_uncoalesced", gpu::WorkSize(workGroupSize, global_work_size / workGroupSize), as, benchmarkingIter);
        execute_sum("compute_sum_coalesced", gpu::WorkSize(workGroupSize, global_work_size / workGroupSize), as, benchmarkingIter);
        execute_sum("compute_sum_using_local_memory", ws, as, benchmarkingIter);
        execute_sum("compute_tree_sum", ws, as, benchmarkingIter);
    }
}
