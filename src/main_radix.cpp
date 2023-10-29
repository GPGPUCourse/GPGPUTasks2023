#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"
#include "cl/matrix_transpose_cl.h"
#include "cl/prefix_sum_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


#define WORK_GROUP_SIZE_MAIN 64
#define WORK_GROUP_SIZE_PREFIX 128
#define WORK_GROUP_SIZE_TRANSPOSE 32
#define COUNT_NUMBER 8
#define BIT_PER_LEVEL 3  // = log_2(COUNT_NUMBER)


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
    unsigned int n = 4096;
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

    gpu::gpu_mem_32u as_gpu, res_gpu, counts_gpu, counts_t_gpu, prefix_res_gpu, prefix_bs_gpu;
    as_gpu.resizeN(n);
    res_gpu.resize(n);

    {
        unsigned int global_work_size_main = (n + WORK_GROUP_SIZE_MAIN - 1) / WORK_GROUP_SIZE_MAIN * WORK_GROUP_SIZE_MAIN;
        counts_gpu.resizeN(global_work_size_main / WORK_GROUP_SIZE_MAIN * COUNT_NUMBER);
        counts_t_gpu.resizeN(global_work_size_main / WORK_GROUP_SIZE_MAIN * COUNT_NUMBER);
        prefix_res_gpu.resizeN(global_work_size_main / WORK_GROUP_SIZE_MAIN * COUNT_NUMBER);
        prefix_bs_gpu.resizeN(global_work_size_main / WORK_GROUP_SIZE_MAIN * COUNT_NUMBER);

        unsigned int global_work_size_0 = (global_work_size_main / WORK_GROUP_SIZE_MAIN + WORK_GROUP_SIZE_TRANSPOSE - 1) / WORK_GROUP_SIZE_TRANSPOSE * WORK_GROUP_SIZE_TRANSPOSE;
        unsigned int global_work_size_1 = (COUNT_NUMBER + WORK_GROUP_SIZE_TRANSPOSE - 1) / WORK_GROUP_SIZE_TRANSPOSE * WORK_GROUP_SIZE_TRANSPOSE;

        unsigned int global_work_size_prefix = (global_work_size_main / WORK_GROUP_SIZE_MAIN * COUNT_NUMBER + WORK_GROUP_SIZE_PREFIX - 1) * WORK_GROUP_SIZE_PREFIX;

        ocl::Kernel counter(radix_kernel, radix_kernel_length, "count");
        counter.compile();
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();
        ocl::Kernel compress(prefix_sum_kernel, prefix_sum_kernel_length, "compress");
        compress.compile();
        ocl::Kernel prefix_sum(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum");
        prefix_sum.compile();
        ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose");
        matrix_transpose_kernel.compile();

        unsigned int loops = 0, k = n;
        while (k > 0) {
            k = k >> BIT_PER_LEVEL;
            ++loops;
        }

        loops = 1;  // DEBUG!!!

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);


            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            
            std::cerr << "Starting" << std::endl;
            for (int level = 0; level < loops; ++level) {
                counter.exec(gpu::WorkSize(WORK_GROUP_SIZE_MAIN, global_work_size_main), as_gpu, counts_gpu, level, n);
                std::cerr << "Got counts" << std::endl;
                // matrix_transpose_kernel.exec(gpu::WorkSize(WORK_GROUP_SIZE_TRANSPOSE, WORK_GROUP_SIZE_TRANSPOSE, global_work_size_1, global_work_size_0), 
                //                              counts_gpu, counts_t_gpu, global_work_size_main / WORK_GROUP_SIZE_MAIN, COUNT_NUMBER);
                // std::cerr << "Transposed!" << std::endl;
                // unsigned int m = global_work_size_main / WORK_GROUP_SIZE_MAIN * COUNT_NUMBER, depth = 0, compress_coef = 2;
                // unsigned int n_compressed = global_work_size_main / WORK_GROUP_SIZE_MAIN * COUNT_NUMBER;
				// for (; m > 0; m /= 2, ++depth, compress_coef *= 2, n_compressed = (n_compressed + 1) / 2) {
				// 	prefix_sum.exec(gpu::WorkSize(WORK_GROUP_SIZE_PREFIX, global_work_size_prefix),
                //                     counts_t_gpu, prefix_res_gpu, global_work_size_main / WORK_GROUP_SIZE_MAIN * COUNT_NUMBER, depth, compress_coef);
				// 	compress.exec(gpu::WorkSize(WORK_GROUP_SIZE_PREFIX, global_work_size_prefix), counts_t_gpu, prefix_bs_gpu, n_compressed);
				// 	counts_t_gpu.swap(prefix_bs_gpu);
				// }
                // std::cerr << "Got prefix sum!!" << std::endl;
                // radix.exec(gpu::WorkSize(WORK_GROUP_SIZE_MAIN, global_work_size_main), as_gpu, res_gpu, prefix_res_gpu, level, n);
                // std::cerr << "!!!" << std::endl;
                // res_gpu.swap(as_gpu);
            }
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        // res_gpu.readN(as.data(), n);
        std::cerr << "CP 1" << std::endl;
        std::vector<unsigned int> cnt(global_work_size_main / WORK_GROUP_SIZE_MAIN * COUNT_NUMBER, 0);
        std::cerr << "CP 2" << std::endl;
        counts_gpu.readN(cnt.data(), 64);

        std::cerr << "Got data" << std::endl;
        for (int j = 0; j < 8; ++j) {
            for (int i = 0; i < COUNT_NUMBER; ++i) {
                std::cerr << cnt[j * COUNT_NUMBER + i] << " ";
            }
            std::cout << std::endl;
        }
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
