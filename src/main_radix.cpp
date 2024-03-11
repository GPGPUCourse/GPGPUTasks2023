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
#include <bitset>


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
    const uint32_t n = 32 * 1024 * 1024;
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

    {
        ocl::Kernel fill_zeros(radix_kernel, radix_kernel_length, "fill_zeros");
        fill_zeros.compile();

        ocl::Kernel local_stable_merge_sort(radix_kernel, radix_kernel_length, "local_stable_merge_sort");
        local_stable_merge_sort.compile();

        ocl::Kernel local_counts(radix_kernel, radix_kernel_length, "local_counts");
        local_counts.compile();

        ocl::Kernel local_counts_transposed(radix_kernel, radix_kernel_length, "local_counts_transposed");
        local_counts_transposed.compile();

        ocl::Kernel prefix_sum(radix_kernel, radix_kernel_length, "prefix_sum");
        prefix_sum.compile();

        ocl::Kernel shift_right(radix_kernel, radix_kernel_length, "shift_right");
        shift_right.compile();

        ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        matrix_transpose.compile();

        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        const int K_bits = 4;
        const int cnt = (1 << K_bits);

        gpu::gpu_mem_32u bs_gpu;
        bs_gpu.resizeN(n);

        const uint32_t workGroupSize = 128;
        const uint32_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        gpu::WorkSize ws(workGroupSize, global_work_size);

        const uint32_t workGroupCnt = global_work_size / workGroupSize;
        const uint32_t matrix_size = workGroupCnt * cnt;

        gpu::WorkSize ws_matrix(workGroupSize, matrix_size);
        gpu::WorkSize ws_transpose(16, 16, cnt, workGroupCnt);

        gpu::gpu_mem_32u local_cnt_matrix;
        local_cnt_matrix.resizeN(matrix_size);

        gpu::gpu_mem_32u local_cnt_matrix_copy;
        local_cnt_matrix_copy.resizeN(matrix_size);

        gpu::gpu_mem_32u local_prefix_sums;
        local_prefix_sums.resizeN(matrix_size);

        gpu::gpu_mem_32u global_prefix_sums;
        global_prefix_sums.resizeN(matrix_size);

        const std::vector<uint32_t> zeroes(matrix_size, 0);
        std::vector<uint32_t> bar(matrix_size, 0);

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                bs_gpu.writeN(std::vector<uint32_t>(n, 0).data(), n);
                int bit_mask = (1 << K_bits) - 1;
                // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
                t.restart();

                for (uint32_t step = 0; step < sizeof(uint32_t) * 8 / K_bits; ++step) {
                    fill_zeros.exec(ws, local_cnt_matrix, matrix_size);
                    fill_zeros.exec(ws, local_cnt_matrix_copy, matrix_size);
                    fill_zeros.exec(ws, local_prefix_sums, matrix_size);
                    fill_zeros.exec(ws, global_prefix_sums, matrix_size);

                    for (uint32_t M = 1; M < workGroupSize; M *= 2) {
                        local_stable_merge_sort.exec(ws, as_gpu, bs_gpu, bit_mask, n, M);
                        as_gpu.swap(bs_gpu);
                    }

                    // Cчетчики для каждой воркгруппы через local_mem и atomic_add, скопировав эту память в глобальную матрицу CNT
                    local_counts.exec(ws, as_gpu, n, local_cnt_matrix, bit_mask, step * K_bits);

                    // Счет по матрице CNT локальных префиксных суммм в каждой воркгруппе
                    local_cnt_matrix.copyToN(local_cnt_matrix_copy, matrix_size);
                    for (uint32_t prefix_offset = 1; prefix_offset <= matrix_size; prefix_offset *= 2) {
                        prefix_sum.exec(ws_matrix, local_cnt_matrix_copy, local_prefix_sums, matrix_size,prefix_offset);
                        local_prefix_sums.swap(local_cnt_matrix_copy);
                    }

                    // Сдвиг префиксов вправо для удобства индексирования
                    shift_right.exec(ws_matrix, local_prefix_sums, local_cnt_matrix_copy, matrix_size);
                    local_prefix_sums.swap(local_cnt_matrix_copy);

                    // Транспонирование матрицы CNT
                    matrix_transpose.exec(ws_transpose, local_cnt_matrix, local_cnt_matrix_copy, workGroupCnt, cnt);

                    // Счет префиксных суммм по матрице CNT
                    for (uint32_t prefix_offset = 1; prefix_offset <= matrix_size; prefix_offset *= 2) {
                        prefix_sum.exec(ws_matrix, local_cnt_matrix_copy, global_prefix_sums, matrix_size,prefix_offset);
                        global_prefix_sums.swap(local_cnt_matrix_copy);
                    }

                    // Сдвиг префиксов вправо для удобства индексирования
                    shift_right.exec(ws_matrix, global_prefix_sums, local_cnt_matrix_copy, matrix_size);
                    global_prefix_sums.swap(local_cnt_matrix_copy);

                    // Запись исходные числа по новым индексам
                    radix.exec(ws, as_gpu, bs_gpu, n, local_prefix_sums, global_prefix_sums, workGroupCnt, bit_mask,step * K_bits);
                    as_gpu.swap(bs_gpu);

                    // Сдвиг маски
                    bit_mask = (bit_mask << K_bits);
                }
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

            as_gpu.readN(as.data(), n);

            // Проверяем корректность результатов
            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
            }
        }
    }

    return 0;
}