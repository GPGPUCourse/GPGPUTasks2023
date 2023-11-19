#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <climits>   // CHAR_BIT
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
    uint n = 32 * 1024 * 1024;
    std::vector<uint> as(n, 0);
    FastRandom r(n);
    for (uint i = 0; i < n; ++i) {
        as[i] = (uint) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<uint> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    // Prepare some constants 
    const uint maskSize = 2;
    const uint uniqueValsCount = (1 << maskSize);
    const uint wgSize = 64;
    const uint cntSize = n / wgSize * uniqueValsCount;

    // Arrays for input & sorted data
    gpu::gpu_mem_32u as_gpu, as_sorted_gpu;
    as_gpu.resizeN(n);
    as_sorted_gpu.resizeN(n);

    // Auxiliary arrays for counters & prefix sums
    gpu::gpu_mem_32u counters_gpu, counters_temp_gpu, prefix_sums_gpu;
    counters_gpu.resizeN(cntSize);
    counters_temp_gpu.resizeN(cntSize);
    prefix_sums_gpu.resizeN(cntSize);

    {
        ocl::Kernel radix_count(radix_kernel, radix_kernel_length, "radix_count");
        radix_count.compile();

        ocl::Kernel prefixSum(radix_kernel, radix_kernel_length, "prefix");
        prefixSum.compile();

        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
        radix_sort.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();

            for (int shiftR = 0; shiftR <= CHAR_BIT * sizeof(decltype(as)::value_type); shiftR += maskSize) {
                radix_count.exec(gpu::WorkSize(wgSize, n), as_gpu, counters_gpu, counters_temp_gpu, shiftR, n);

                // Do the sorting
				for (uint blockSize = 1; blockSize <= cntSize; blockSize *= 2) {
					prefixSum.exec(gpu::WorkSize(wgSize, cntSize), counters_temp_gpu, prefix_sums_gpu, blockSize, n);
					counters_temp_gpu.swap(prefix_sums_gpu);
				}
                counters_temp_gpu.swap(prefix_sums_gpu);

                radix_sort.exec(gpu::WorkSize(wgSize, n), as_gpu, as_sorted_gpu,
                                prefix_sums_gpu, counters_gpu, shiftR, n);
                as_sorted_gpu.swap(as_gpu);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_sorted_gpu.readN(as.data(), n);

        // Проверяем корректность результатов
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
        }

        return 0;
    }
}