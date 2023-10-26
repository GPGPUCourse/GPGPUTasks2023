#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


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
    unsigned n = 1024 * 1024;
    std::vector<unsigned> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned> cpu_sorted = as;
    std::sort(cpu_sorted.begin(), cpu_sorted.end());
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            std::vector<unsigned> as_cpu = as;
            t.restart();
            std::sort(as_cpu.begin(), as_cpu.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n * 1e-6) / t.lapAvg() << " millions/s" << std::endl;
    }

    auto examineGPUSortAlgo = [&](std::string algoName, auto sort) {
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();
            sort(as_gpu);
            t.nextLap();
        }
        std::cout << "GPU " << algoName << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " << algoName << ": " << (n * 1e-6) / t.lapAvg() << " millions/s" << std::endl;
        std::vector<unsigned> result(n);
        as_gpu.readN(result.data(), n);
        for (size_t i = 0; i < n; ++i)
            EXPECT_THE_SAME(result[i], cpu_sorted[i], "GPU " + algoName + " results should be correct!");
    };

    ocl::Kernel bitonic(bitonic_kernel, bitonic_kernel_length, "bitonicSortStep");
    bitonic.compile();
    auto bitonicSort = [&](gpu::gpu_mem_32u &as_gpu) {
        for (int blockToSortSize = 2; blockToSortSize <= n; blockToSortSize *= 2) {
            for (int slidingBlockSize = blockToSortSize; slidingBlockSize > 1; slidingBlockSize /= 2) {
                bitonic.exec(gpu::WorkSize(128, n / 2), as_gpu, blockToSortSize, slidingBlockSize);
            }
        }
    };
    examineGPUSortAlgo("bitonic", bitonicSort);

    return 0;
}
