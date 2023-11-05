#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"
#include "cl/matrix_transpose_cl.h"
#include "cl/prefix_sum_cl.h"
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
    gpu::gpu_mem_32u as_gpu, cnts_gpu, cntst_gpu, pref_cnts_gpu, pref_cnt_gpu, res_gpu, res_sort_gpu;
    as_gpu.resizeN(n);
    cnts_gpu.resizeN(n);
    cntst_gpu.resizeN(n);
    pref_cnts_gpu.resizeN(n);
    pref_cnt_gpu.resizeN(n);
    res_gpu.resizeN(n);
    res_sort_gpu.resizeN(n);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();
        ocl::Kernel count(radix_kernel, radix_kernel_length, "get_counts");
        count.compile();
        ocl::Kernel transpose(matrix_transpose_kernel, matrix_transpose_kernel_length, "matrix_transpose");
        transpose.compile();
        ocl::Kernel prefix(prefix_sum_kernel, prefix_sum_kernel_length, "prefix");
        prefix.compile();
        ocl::Kernel prefix_row(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_row");
        prefix_row.compile();
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge");
        merge.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (int offset = 0; offset < 32; offset += 4) {
                for (int block_size = 1; block_size < 128; block_size *= 2) {
                    merge.exec(gpu::WorkSize(128, n), as_gpu, res_sort_gpu, block_size, offset);
                    as_gpu.swap(res_sort_gpu);
                }

                count.exec(gpu::WorkSize(128, n), as_gpu, cnts_gpu, offset);

                auto transpose_ws = gpu::WorkSize(16, 16, 16, n / 128);
                transpose.exec(transpose_ws, cnts_gpu, cntst_gpu, n / 128, 16);

                for (int off = 1; off < n; off *= 2) {
                    prefix.exec(gpu::WorkSize(128, n), cntst_gpu, pref_cnts_gpu, off);
                    cntst_gpu.swap(pref_cnts_gpu);
                }
                cntst_gpu.swap(pref_cnts_gpu);

                prefix_row.exec(gpu::WorkSize(128, n), cnts_gpu, pref_cnt_gpu);

                radix.exec(gpu::WorkSize(128, n), as_gpu, res_gpu, pref_cnt_gpu,
                           pref_cnts_gpu, n, offset);

                res_gpu.swap(as_gpu);
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
