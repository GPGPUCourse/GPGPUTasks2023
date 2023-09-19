#include "cl/sum_cl.h"
#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <cassert>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


void execKernel(ocl::Kernel &kernel, gpu::gpu_mem_32u const &aGpu, gpu::gpu_mem_32u &resGPU, uint workGroupSize,
                uint globalWorkSize, uint n, uint expectedSum, uint benchmarkingIters) {
    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        uint actualSum = 0;
        resGPU.writeN(&actualSum, 1);
        kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), aGpu, resGPU, n);
        resGPU.readN(&actualSum, 1);
        EXPECT_THE_SAME(expectedSum, actualSum, "GPU (" + kernel.getKernelName() + ") result should be consistent!");
        t.nextLap();
    }

    std::cout << "GPU (" << kernel.getKernelName() << "):     " << t.lapAvg() << "+-" << t.lapStd() << " s"
              << std::endl;
    std::cout << "GPU (" << kernel.getKernelName() << "):     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s"
              << std::endl;
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
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
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+ : sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u aGpu;
        gpu::gpu_mem_32u resGPU;
        unsigned int workGroupSize = 64;
        unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        aGpu.resizeN(globalWorkSize);
        as.resize(globalWorkSize);
        resGPU.resizeN(1);
        aGpu.writeN(as.data(), n);
        {
            ocl::Kernel atomicAddSum(sum_kernel, sum_kernel_length, "atomicAddSum");
            execKernel(atomicAddSum, aGpu, resGPU, workGroupSize, globalWorkSize, as.size(), reference_sum,
                       benchmarkingIters);
        }

        uint valuesPerItem = 32;
        {
            assert(globalWorkSize % valuesPerItem == 0);
            ocl::Kernel loopSum(sum_kernel, sum_kernel_length, "loopSum");
            execKernel(loopSum, aGpu, resGPU, workGroupSize, globalWorkSize / valuesPerItem, as.size(), reference_sum,
                       benchmarkingIters);
        }
        {
            ocl::Kernel loopCoalesedSum(sum_kernel, sum_kernel_length, "loopCoalesedSum");
            execKernel(loopCoalesedSum, aGpu, resGPU, workGroupSize, globalWorkSize / valuesPerItem, as.size(),
                       reference_sum, benchmarkingIters);
        }

        {
            uint smallValuesPerItem = 4;
            assert(globalWorkSize % smallValuesPerItem == 0);
            ocl::Kernel smallLoopSum(sum_kernel, sum_kernel_length, "smallLoopSum");
            execKernel(smallLoopSum, aGpu, resGPU, workGroupSize, globalWorkSize / smallValuesPerItem, as.size(),
                       reference_sum, benchmarkingIters);
        }

        {
            ocl::Kernel localMemSum(sum_kernel, sum_kernel_length, "localMemSum");
            execKernel(localMemSum, aGpu, resGPU, workGroupSize, globalWorkSize, as.size(), reference_sum,
                       benchmarkingIters);
        }

                {
            ocl::Kernel treeSum(sum_kernel, sum_kernel_length, "treeSum");
            execKernel(treeSum, aGpu, resGPU, workGroupSize, globalWorkSize, as.size(), reference_sum,
                       benchmarkingIters);
        }
    }
}
