#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
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
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
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

    {
        gpu::gpu_mem_32u as_gpu, bs_gpu, counters_gpu, counters_pref_gpu, counters_res_gpu;
        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);

        unsigned int global_block_size = n / 128;

        counters_gpu.resizeN(global_block_size);
        counters_pref_gpu.resizeN(global_block_size);
        counters_res_gpu.resizeN(global_block_size);

        std::vector<unsigned int> counters_pref(n, 0);

        ocl::Kernel reduce(radix_kernel, radix_kernel_length, "reduce");
        ocl::Kernel prefix_sum(radix_kernel, radix_kernel_length, "prefix_sum");
        ocl::Kernel counters(radix_kernel, radix_kernel_length, "cnt_calc");
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");

        reduce.compile();
        prefix_sum.compile();
        counters.compile();
        radix.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            bs_gpu.writeN(counters_pref.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (global_block_size + workGroupSize - 1) / workGroupSize * workGroupSize;
            unsigned int total_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            for (unsigned int current_bit = 0; current_bit < 32; current_bit++) {
                counters.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, counters_gpu, current_bit, n / 128);
                counters_pref_gpu.writeN(counters_pref.data(), global_block_size);

                for (unsigned int block_size = 1; block_size <= global_block_size; block_size *= 2) {
                    prefix_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), counters_gpu, counters_pref_gpu, global_block_size, block_size);
                    reduce.exec(gpu::WorkSize(workGroupSize, global_work_size), counters_gpu, counters_res_gpu, global_block_size / (block_size * 2));
                    counters_gpu.swap(counters_res_gpu);
                }

                radix.exec(gpu::WorkSize(workGroupSize, total_work_size), counters_pref_gpu, as_gpu, bs_gpu,
                           current_bit, n);
                as_gpu.swap(bs_gpu);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}