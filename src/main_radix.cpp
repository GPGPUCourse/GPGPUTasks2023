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

    unsigned int bits = 2;
    unsigned int values = (1 << bits);
    unsigned int work_group_size = 128;
    unsigned int small_work_group_size = 16;

    gpu::gpu_mem_32u as_gpu, counter_gpu, prefix_part_sums_gpu, prefix_result_gpu, tmp_gpu;
    as_gpu.resizeN(n);
    counter_gpu.resizeN(n / work_group_size * values);
    prefix_part_sums_gpu.resizeN(n / work_group_size * values);
    prefix_result_gpu.resizeN(n / work_group_size * values + 1);
    tmp_gpu.resizeN(n);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        ocl::Kernel counter(radix_kernel, radix_kernel_length, "counter");
        ocl::Kernel calculate_parts(radix_kernel, radix_kernel_length, "calculate_prefix_parts");
        ocl::Kernel calculate_prefix_sums(radix_kernel, radix_kernel_length, "calculate_prefix_sums");
        ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        ocl::Kernel reset(radix_kernel, radix_kernel_length, "reset");
        radix.compile();
        counter.compile();
        calculate_parts.compile();
        calculate_prefix_sums.compile();
        matrix_transpose.compile();
        reset.compile();
        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);

                t.nextLap();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
                for (int bit_group = 0; bit_group * bits < 32; bit_group++) {
                    std::vector<unsigned> tmp(n);
                    counter.exec(gpu::WorkSize(work_group_size, n), as_gpu, counter_gpu, bit_group);

                    matrix_transpose.exec(gpu::WorkSize(small_work_group_size, small_work_group_size,
                                                        (values + small_work_group_size - 1) / small_work_group_size *
                                                                small_work_group_size,
                                                        (n / work_group_size + small_work_group_size - 1) /
                                                                small_work_group_size * small_work_group_size),
                                          counter_gpu, prefix_part_sums_gpu, values, n / work_group_size);

                    unsigned int tmp_n = n / work_group_size * values;

                    for (int size = 1; size <= tmp_n; size *= 2) {
                        if (size != 1)
                            calculate_parts.exec(gpu::WorkSize(std::min(work_group_size, tmp_n / size), tmp_n / size),
                                                 prefix_part_sums_gpu, size);
                        calculate_prefix_sums.exec(gpu::WorkSize(work_group_size, tmp_n), prefix_result_gpu,
                                                   prefix_part_sums_gpu, size);
                    }

                    unsigned x = 0;
                    prefix_result_gpu.writeN(&x, 1);

                    radix.exec(gpu::WorkSize(work_group_size, n), as_gpu, prefix_result_gpu, tmp_gpu, bit_group,
                               n / work_group_size);

                    reset.exec(gpu::WorkSize(work_group_size, tmp_n), prefix_result_gpu);
                    std::swap(as_gpu, tmp_gpu);
                }
            }
            t.stop();
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
