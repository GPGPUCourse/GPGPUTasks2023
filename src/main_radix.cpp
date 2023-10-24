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

    int benchmarkingIters = 1;
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
        gpu::gpu_mem_32u as_gpu, bs_gpu, counters_gpu, counters_bs_gpu, counters_cs_gpu;
        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);
        unsigned int counter_size = n / 128;
        counters_gpu.resizeN(counter_size);
        counters_bs_gpu.resizeN(counter_size);
        counters_cs_gpu.resizeN(counter_size);
        std::vector<unsigned int> zeros(n, 0);

        ocl::Kernel prefix_sum_reduce(radix_kernel, radix_kernel_length, "reduce");
        prefix_sum_reduce.compile();
        ocl::Kernel prefix_sum_gather(radix_kernel, radix_kernel_length, "prefix_sum");
        prefix_sum_gather.compile();
        ocl::Kernel radix_counters(radix_kernel, radix_kernel_length, "radix_counters");
        radix_counters.compile();
        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
        radix_sort.compile();

        timer t;
        for (int iter = 0; iter < 1; ++iter) {
            as_gpu.writeN(as.data(), n);
            bs_gpu.writeN(zeros.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (unsigned int offset = 0; offset < 32; ++offset) {
                unsigned int workGroupSize = 128;
                unsigned int global_work_size = (counter_size + workGroupSize - 1) / workGroupSize * workGroupSize;
                radix_counters.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, counters_gpu, offset, n);
                counters_bs_gpu.writeN(zeros.data(), counter_size);

                for (unsigned int level = 0; (1 << level) <= counter_size; ++level) {
                    prefix_sum_reduce.exec(gpu::WorkSize(workGroupSize, global_work_size), counters_gpu,
                                           counters_bs_gpu, counter_size, level);
                    prefix_sum_gather.exec(gpu::WorkSize(workGroupSize, global_work_size), counters_gpu,
                                           counters_cs_gpu, counter_size / (1 << (level + 1)));
                    counters_gpu.swap(counters_cs_gpu);
                }

                global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
                radix_sort.exec(gpu::WorkSize(workGroupSize, global_work_size), counters_bs_gpu, as_gpu, bs_gpu, offset,
                                n);
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
