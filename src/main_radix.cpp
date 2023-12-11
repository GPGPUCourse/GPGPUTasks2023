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
#define LOG_MAX_DIGIT 5
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

    const int NUM_WORKGROUPS = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
    const int NUM_SORTS = (32 + LOG_MAX_DIGIT - 1) / LOG_MAX_DIGIT;
    static_assert(2 * MAX_DIGIT <= WORKGROUP_SIZE);

    std::string radix_kernel_defines = " -D WORKGROUP_SIZE=" + to_string(WORKGROUP_SIZE) +
                                       " -D LOG_MAX_DIGIT=" + to_string(LOG_MAX_DIGIT) +
                                       " -D MAX_DIGIT=" + to_string(MAX_DIGIT);

    // VERSION 1
    if (1)
    {
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        gpu::gpu_mem_32u bs_gpu;
        bs_gpu.resizeN(n);
        gpu::gpu_mem_32u t_gpu;
        t_gpu.resizeN(NUM_WORKGROUPS * MAX_DIGIT);
        std::vector<unsigned int> digits_less(MAX_DIGIT, 0);
        std::vector<unsigned int> tmp(MAX_DIGIT, 0);
        gpu::gpu_mem_32u digits_less_gpu;
        digits_less_gpu.resizeN(MAX_DIGIT);

        ocl::Kernel fill0(radix_kernel, radix_kernel_length, "fill0", radix_kernel_defines);
        fill0.compile();
        ocl::Kernel counts(radix_kernel, radix_kernel_length, "counts", radix_kernel_defines);
        counts.compile();
        ocl::Kernel sums(radix_kernel, radix_kernel_length, "sums", radix_kernel_defines);
        sums.compile();
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix", radix_kernel_defines);
        radix.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (int d = 0; d < NUM_SORTS; d++) {
                fill0.exec(gpu::WorkSize(WORKGROUP_SIZE,
                                         (NUM_WORKGROUPS * MAX_DIGIT + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE *
                                         WORKGROUP_SIZE),
                           t_gpu, NUM_WORKGROUPS * MAX_DIGIT);
                counts.exec(gpu::WorkSize(WORKGROUP_SIZE, NUM_WORKGROUPS * WORKGROUP_SIZE),
                            as_gpu, t_gpu, d, n);

                for (int len = 2; len <= n; len <<= 1) {
                    sums.exec(gpu::WorkSize(WORKGROUP_SIZE,
                                            ((NUM_WORKGROUPS + len - 1) / len + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE *
                                            WORKGROUP_SIZE),
                              t_gpu, len, NUM_WORKGROUPS);
                }

                // compute the amount of each digit on cpu
                digits_less.assign(MAX_DIGIT, 0);
                int pnt = NUM_WORKGROUPS - 1;
                while (pnt >= 0) {
                    t_gpu.readN(tmp.data(), MAX_DIGIT, pnt * MAX_DIGIT);
                    for (int x = 0; x + 1 < MAX_DIGIT; x++) {
                        digits_less[x + 1] += tmp[x];
                    }
                    pnt &= (pnt + 1);
                    pnt--;
                }
                for (int x = 1; x < MAX_DIGIT; x++) {
                    digits_less[x] += digits_less[x - 1];
                }
                digits_less_gpu.writeN(digits_less.data(), MAX_DIGIT);

                radix.exec(gpu::WorkSize(WORKGROUP_SIZE, NUM_WORKGROUPS * WORKGROUP_SIZE),
                           as_gpu, bs_gpu, t_gpu, d, n, digits_less_gpu);
                std::swap(as_gpu, bs_gpu);
            }

            t.nextLap();
        }
        std::cout << "GPU (VERSION 1): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU (VERSION 1): " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        std::vector<unsigned int> gpu_sorted(n);
        as_gpu.readN(gpu_sorted.data(), n);

        // Проверяем корректность результатов
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(gpu_sorted[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
        }
    }

    // VERSION 2
    if (1)
    {
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        gpu::gpu_mem_32u bs_gpu;
        bs_gpu.resizeN(n);
        gpu::gpu_mem_32u t_gpu;
        t_gpu.resizeN(NUM_WORKGROUPS * MAX_DIGIT);
        std::vector<unsigned int> digits_less(MAX_DIGIT, 0);
        std::vector<unsigned int> tmp(MAX_DIGIT, 0);
        gpu::gpu_mem_32u digits_less_gpu;
        digits_less_gpu.resizeN(MAX_DIGIT);

        ocl::Kernel counts2(radix_kernel, radix_kernel_length, "counts2", radix_kernel_defines);
        counts2.compile();
        ocl::Kernel sums(radix_kernel, radix_kernel_length, "sums", radix_kernel_defines);
        sums.compile();
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix", radix_kernel_defines);
        radix.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (int d = 0; d < NUM_SORTS; d++) {
                counts2.exec(gpu::WorkSize(WORKGROUP_SIZE, NUM_WORKGROUPS * WORKGROUP_SIZE),
                            as_gpu, t_gpu, d, n);

                for (int len = 2; len <= n; len <<= 1) {
                    sums.exec(gpu::WorkSize(WORKGROUP_SIZE,
                                            ((NUM_WORKGROUPS + len - 1) / len + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE *
                                            WORKGROUP_SIZE),
                              t_gpu, len, NUM_WORKGROUPS);
                }

                // compute the amount of each digit on cpu
                digits_less.assign(MAX_DIGIT, 0);
                int pnt = NUM_WORKGROUPS - 1;
                while (pnt >= 0) {
                    t_gpu.readN(tmp.data(), MAX_DIGIT, pnt * MAX_DIGIT);
                    for (int x = 0; x + 1 < MAX_DIGIT; x++) {
                        digits_less[x + 1] += tmp[x];
                    }
                    pnt &= (pnt + 1);
                    pnt--;
                }
                for (int x = 1; x < MAX_DIGIT; x++) {
                    digits_less[x] += digits_less[x - 1];
                }
                digits_less_gpu.writeN(digits_less.data(), MAX_DIGIT);

                radix.exec(gpu::WorkSize(WORKGROUP_SIZE, NUM_WORKGROUPS * WORKGROUP_SIZE),
                           as_gpu, bs_gpu, t_gpu, d, n, digits_less_gpu);
                std::swap(as_gpu, bs_gpu);
            }

            t.nextLap();
        }
        std::cout << "GPU (VERSION 2): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU (VERSION 2): " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        std::vector<unsigned int> gpu_sorted(n);
        as_gpu.readN(gpu_sorted.data(), n);

        // Проверяем корректность результатов
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(gpu_sorted[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
        }
    }

    return 0;
}
