#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/sum_cl.h"

#include <cassert>
#include <functional>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

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

    auto benchmarkGPU = [&](const std::function<unsigned()> &RunCallback, std::string kernelName) {
        timer timer;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned sum = RunCallback();
            EXPECT_THE_SAME(reference_sum, sum, "GPU " + kernelName + " result should be consistent!");
            timer.nextLap();
        }
        std::cout << "GPU " << kernelName << ": " << timer.lapAvg() << "+-" << timer.lapStd() << " s" << std::endl;
        std::cout << "GPU " << kernelName << ": " << (n / 1000.0 / 1000.0) / timer.lapAvg() << " millions/s"
                  << std::endl;
    };


    gpu::gpu_mem_32u array_gpu;
    array_gpu.resizeN(n);
    array_gpu.writeN(as.data(), n);
    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_global_atomic");
        kernel.compile(/* printLog= */ false);
        unsigned workGroupSize = 32;
        unsigned workSpaceSize = gpu::divup(n, workGroupSize) * workGroupSize;
        gpu::WorkSize workSize(workGroupSize, workSpaceSize);
        gpu::gpu_mem_32u result_gpu;
        result_gpu.resizeN(1);
        unsigned zero = 0;
        auto callback = [&]() -> unsigned {
            result_gpu.writeN(&zero, 1);
            kernel.exec(workSize, array_gpu, n, result_gpu);
            unsigned result;
            result_gpu.readN(&result, 1);
            return result;
        };
        benchmarkGPU(callback, /* kernelName = */ "sum_global_atomic");
    }
    {
        static constexpr unsigned valuesPerWorkitem = 128;
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_noncoalesced_loop");
        kernel.compile(/* printLog= */ false);
        unsigned workGroupSize = 8;
        unsigned workSpaceSize = gpu::divup(n, valuesPerWorkitem);
        gpu::WorkSize workSize(workGroupSize, workSpaceSize);
        gpu::gpu_mem_32u result_gpu;
        result_gpu.resizeN(1);
        unsigned zero = 0;
        auto callback = [&]() -> unsigned {
            result_gpu.writeN(&zero, 1);
            kernel.exec(workSize, array_gpu, n, result_gpu);
            unsigned result;
            result_gpu.readN(&result, 1);
            return result;
        };
        benchmarkGPU(callback, /* kernelName = */ "sum_noncoalesced_loop");
    }
    {
        static constexpr unsigned valuesPerWorkitem = 128;
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_coalesced_loop");
        kernel.compile(/* printLog= */ false);
        unsigned workGroupSize = 8;
        unsigned workSpaceSize = gpu::divup(n, workGroupSize * valuesPerWorkitem) * workGroupSize;
        gpu::WorkSize workSize(workGroupSize, workSpaceSize);
        gpu::gpu_mem_32u result_gpu;
        result_gpu.resizeN(1);
        unsigned zero = 0;
        auto callback = [&]() -> unsigned {
            result_gpu.writeN(&zero, 1);
            kernel.exec(workSize, array_gpu, n, result_gpu);
            unsigned result;
            result_gpu.readN(&result, 1);
            return result;
        };
        benchmarkGPU(callback, /* kernelName = */ "sum_coalesced_loop");
    }
    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_local_copy");
        kernel.compile(/* printLog= */ false);
        static constexpr unsigned workGroupSize = 1024;
        unsigned workSpaceSize = gpu::divup(n, workGroupSize) * workGroupSize;
        gpu::WorkSize workSize(workGroupSize, workSpaceSize);
        gpu::gpu_mem_32u result_gpu;
        result_gpu.resizeN(1);
        unsigned zero = 0;
        auto callback = [&]() -> unsigned {
            result_gpu.writeN(&zero, 1);
            kernel.exec(workSize, array_gpu, n, result_gpu);
            unsigned result;
            result_gpu.readN(&result, 1);
            return result;
        };
        benchmarkGPU(callback, /* kernelName = */ "sum_local_copy");
    }
    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_tree");
        kernel.compile(/* printLog= */ false);
        static constexpr unsigned workGroupSize = 512;
        unsigned workSpaceSize = gpu::divup(n, workGroupSize) * workGroupSize;
        gpu::WorkSize workSize(workGroupSize, workSpaceSize);
        gpu::gpu_mem_32u result_gpu;
        result_gpu.resizeN(1);
        unsigned zero = 0;
        auto callback = [&]() -> unsigned {
            result_gpu.writeN(&zero, 1);
            kernel.exec(workSize, array_gpu, n, result_gpu);
            unsigned result;
            result_gpu.readN(&result, 1);
            return result;
        };
        benchmarkGPU(callback, /* kernelName = */ "sum_tree");
    }
}
