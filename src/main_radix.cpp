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
#define WORKGROUP_SIZE 128
#define LOG_MAX_DIGIT 4
#define MAX_DIGIT (1 << LOG_MAX_DIGIT)

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

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);
    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(n);
    gpu::gpu_mem_32u t_gpu;
    t_gpu.resizeN(MAX_DIGIT * n);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix",
                           " -D WORKGROUP_SIZE=" + to_string(WORKGROUP_SIZE) +
                                  " -D LOG_MAX_DIGIT=" + to_string(LOG_MAX_DIGIT) +
                                  " -D MAX_DIGIT=" + to_string(MAX_DIGIT));
        radix.compile();
        ocl::Kernel sums1(radix_kernel, radix_kernel_length, "sums1",
                           " -D WORKGROUP_SIZE=" + to_string(WORKGROUP_SIZE) +
                                  " -D LOG_MAX_DIGIT=" + to_string(LOG_MAX_DIGIT) +
                                  " -D MAX_DIGIT=" + to_string(MAX_DIGIT));
        sums1.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            unsigned int all_work1 = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;
            for (int d = 0; d < 8; d++)
            {
                for (int len = 1; len < 2 * n; len <<= 1)
                {
                    unsigned int all_work2 = (n / len + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;
                    sums1.exec(gpu::WorkSize(WORKGROUP_SIZE, all_work2), as_gpu,t_gpu, d,len, n);
                }
                radix.exec(gpu::WorkSize(WORKGROUP_SIZE, all_work1), as_gpu, bs_gpu, t_gpu, d,n);
                std::swap(as_gpu, bs_gpu);
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
