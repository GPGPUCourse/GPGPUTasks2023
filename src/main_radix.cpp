#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/matrix_transpose_cl.h"
#include "cl/prefix_sum_cl.h"
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

    const int benchmarkingIters = 10;
    const unsigned int n = 32 * 1024 * 1024;
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
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);

        gpu::gpu_mem_32u bs_gpu;
        bs_gpu.resizeN(n);

        const unsigned int work_group_size = 128;
        const unsigned int work_groups_count = n / 128;

        const unsigned int bits_per_iter = 4;
        const unsigned int bit_iters = 32 / bits_per_iter;

        const unsigned int cnt_row_size = 1 << bits_per_iter;
        const unsigned int cnt_size = work_groups_count * cnt_row_size;

        gpu::gpu_mem_32u cnt_gpu;
        cnt_gpu.resizeN(cnt_size);

        gpu::gpu_mem_32u cnt_prefix_gpu;
        cnt_prefix_gpu.resizeN(cnt_size);

        ocl::Kernel radix_count(radix_kernel, radix_kernel_length, "radix_count");
        radix_count.compile();

        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
        radix_sort.compile();

        ocl::Kernel matrix_transpose(matrix_transpose_kernel, matrix_transpose_kernel_length, "matrix_transpose");
        matrix_transpose.compile();

        ocl::Kernel prefix_sum(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum");
        prefix_sum.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            // Запускаем секундомер после прогрузки данных,
            // чтобы замерять время работы кернела, а не трансфер данных
            t.restart();

            for (unsigned int iter = 0; iter < bit_iters; ++iter) {

                // подсчитываем количество
                radix_count.exec(gpu::WorkSize(work_group_size, n), as_gpu, cnt_gpu, iter);

                // транспонируем матрицу
                matrix_transpose.exec(
                    gpu::WorkSize(16, 16, cnt_row_size, work_groups_count),
                    cnt_gpu, cnt_prefix_gpu, cnt_row_size, work_groups_count
                );

                // подсчитываем префикс сумму
                for (unsigned int mask = 1; mask < cnt_size; mask <<= 1) {
                    prefix_sum.exec(gpu::WorkSize(work_group_size, cnt_size / 2), cnt_prefix_gpu, mask);
                }

                // размещаем значения по своим местам
                radix_sort.exec(gpu::WorkSize(work_group_size, n), as_gpu, cnt_gpu, cnt_prefix_gpu, bs_gpu, iter);

                // сохраняем результат
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
