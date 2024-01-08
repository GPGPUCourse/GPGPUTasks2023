#include "cl/sum_cl.h"

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
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


int main(int argc, char **argv) {
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

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    {
        // TODO: implement on OpenCL
        // gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::gpu_mem_32u array_gpu;
        array_gpu.resizeN(n);
        array_gpu.writeN(as.data(), n);

        gpu::gpu_mem_32u sum_gpu;
        sum_gpu.resizeN(1);

        unsigned sum;
        const unsigned zero = 0;

        // с глобальным атомарным добавлением
        {
            std::string kernelName = "sum_global_atomic_add";
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernelName);
            bool printLog = false;
            kernel.compile(printLog);

            unsigned workGroupSize = 256;
            unsigned workSizeGlobal = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int i = 0; i < benchmarkingIters; ++i) {
                sum_gpu.writeN(&zero, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, workSizeGlobal), array_gpu, n, sum_gpu);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU " + kernelName + " result should be consistent!");
                t.nextLap();
            }

            std::cout << "GPU (" << kernelName << "): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (" << kernelName << "): " << (n / 1e6) / t.lapAvg() << " millions/s\n" << std::endl;
        }

        // с циклом
        {
            std::string kernelName = "sum_noncoalesced_loop";
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernelName);
            bool printLog = false;
            kernel.compile(printLog);

            unsigned workItemVals = 128;
            unsigned workGroupSize = 256;
            unsigned workitemsNum = (n + workItemVals - 1) / workItemVals;
            unsigned workSizeGlobal = (workitemsNum + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int i = 0; i < benchmarkingIters; ++i) {
                sum_gpu.writeN(&zero, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, workSizeGlobal), array_gpu, n, sum_gpu);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU " + kernelName + " result should be consistent!");
                t.nextLap();
            }

            std::cout << "GPU (" << kernelName << "): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (" << kernelName << "): " << (n / 1e6) / t.lapAvg() << " millions/s\n" << std::endl;
        }

        // с циклом и coalesced доступом
        {
            std::string kernelName = "sum_coalesced_loop";
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernelName);
            bool printLog = false;
            kernel.compile(printLog);

            unsigned workItemVals = 128;
            unsigned workGroupSize = 256;
            unsigned workitemsNum = (n + workItemVals - 1) / workItemVals;
            unsigned workSizeGlobal = (workitemsNum + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int i = 0; i < benchmarkingIters; ++i) {
                sum_gpu.writeN(&zero, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, workSizeGlobal), array_gpu, n, sum_gpu);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU " + kernelName + " result should be consistent!");
                t.nextLap();
            }

            std::cout << "GPU (" << kernelName << "): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (" << kernelName << "): " << (n / 1e6) / t.lapAvg() << " millions/s\n" << std::endl;
        }

        // с локальной памятью и главным потоком
        {
            std::string kernelName = "sum_local";
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernelName);
            bool printLog = false;
            kernel.compile(printLog);

            unsigned workGroupSize = 256;
            unsigned workSizeGlobal = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int i = 0; i < benchmarkingIters; ++i) {
                sum_gpu.writeN(&zero, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, workSizeGlobal), array_gpu, n, sum_gpu);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU " + kernelName + " result should be consistent!");
                t.nextLap();
            }

            std::cout << "GPU (" << kernelName << "): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (" << kernelName << "): " << (n / 1e6) / t.lapAvg() << " millions/s\n" << std::endl;
        }

        // с деревом
        {
            std::string kernelName = "sum_tree";
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernelName);
            bool printLog = false;
            kernel.compile(printLog);

            unsigned workGroupSize = 256;
            unsigned workSizeGlobal = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int i = 0; i < benchmarkingIters; ++i) {
                sum_gpu.writeN(&zero, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, workSizeGlobal), array_gpu, n, sum_gpu);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU " + kernelName + " result should be consistent!");
                t.nextLap();
            }

            std::cout << "GPU (" << kernelName << "): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (" << kernelName << "): " << (n / 1e6) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
