#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

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


static unsigned int power_of_two(unsigned int n) {
    unsigned int x = n & (~n + 1);

    while (x < n) {
        x <<= 1;
    }

    return x;
}


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    const int benchmarkingIters = 10;
    const unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
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

    const unsigned int good_n = power_of_two(n);
    as.resize(good_n, 1.0 / 0.0);

    gpu::gpu_mem_32f as_gpu, bs_gpu;
    as_gpu.resizeN(good_n);
    bs_gpu.resizeN(good_n);

    {
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge_naive");
        merge.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), good_n);

            const unsigned int workGroupSize = 128;

            // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных
            t.restart();

            for (unsigned int k = 1; k <= good_n / 2; k <<= 1) {
                const unsigned int gws = good_n / 2 / k;

                merge.exec(gpu::WorkSize(std::min(workGroupSize, gws), gws), as_gpu, bs_gpu, k);

                bs_gpu.copyToN(as_gpu, n);
            }

            t.nextLap();
        }

        std::cout << "GPU naïve: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU naïve: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);

        // Проверяем корректность результатов
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
        }
    }

    // {
    //     ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge");
    //     merge.compile();
    //     timer t;
    //     for (int iter = 0; iter < benchmarkingIters; ++iter) {
    //         as_gpu.writeN(as.data(), n);

    //         // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных
    //         t.restart();

    //         const unsigned int workGroupSize = 128;
    //         const unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    //         merge.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, n);

    //         t.nextLap();
    //     }
    //     std::cout << "GPU merge: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    //     std::cout << "GPU merge: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    //     as_gpu.readN(as.data(), n);

    //     // Проверяем корректность результатов
    //     for (int i = 0; i < n; ++i) {
    //         EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    //     }
    // }

    return 0;
}
