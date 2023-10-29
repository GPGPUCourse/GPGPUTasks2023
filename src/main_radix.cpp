#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"
#include "cl/matrix_transpose.cl.h"
#include "cl/prefix_sum_cl.h"

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

    unsigned int work_size = 128;
    unsigned int work_group_cnt = n / work_size;

    unsigned int total_bit = 32;
    unsigned int bit_per_iter = 4;

    unsigned int numbers_per_cnt_group = 1 << bit_per_iter;
    unsigned int total_counting_size = numbers_per_cnt_group * work_group_cnt;

    gpu::gpu_mem_32u counting_gpu;
    counting_gpu.resizeN(total_counting_size);

    gpu::gpu_mem_32u prefix_sum_gpu;
    prefix_sum_gpu.resizeN(total_counting_size);

    gpu::gpu_mem_32u prefix_sum_tmp_gpu;
    prefix_sum_tmp_gpu.resizeN(total_counting_size);

    gpu::gpu_mem_32u as_gpu, bs_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);

    std::vector<unsigned int> test(total_counting_size, 0);
    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        ocl::Kernel radix_count(radix_kernel, radix_kernel_length, "radix_count");
        radix_count.compile();

        ocl::Kernel matrix_transpose(matrix_transpose_kernel, matrix_transpose_kernel_length, "matrix_transpose");
        matrix_transpose.compile();

        ocl::Kernel prefix_sum(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum");
        prefix_sum.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            for (unsigned int shift = 0; shift < total_bit; shift += bit_per_iter) {

                radix_count.exec(gpu::WorkSize(work_size, n), as_gpu, counting_gpu, shift);

                unsigned int wg_n = 16;
                matrix_transpose.exec(gpu::WorkSize(wg_n, wg_n, numbers_per_cnt_group, work_group_cnt), counting_gpu, prefix_sum_gpu, numbers_per_cnt_group, work_group_cnt);

                for (unsigned int k = 1; k < total_counting_size; k <<= 1) {
                    prefix_sum.exec(gpu::WorkSize(work_size, total_counting_size), prefix_sum_gpu, prefix_sum_tmp_gpu, k);
                    prefix_sum_tmp_gpu.swap(prefix_sum_gpu);
                }

                radix.exec(gpu::WorkSize(work_size, n), as_gpu, bs_gpu, prefix_sum_gpu, counting_gpu, shift);
                bs_gpu.swap(as_gpu);
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
