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

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel radix_count(radix_kernel, radix_kernel_length, "radix_count");
        radix_count.compile();

        ocl::Kernel prefix_sum_up(radix_kernel, radix_kernel_length, "prefix_sum_up");
        prefix_sum_up.compile();

        ocl::Kernel prefix_sum_down(radix_kernel, radix_kernel_length, "prefix_sum_down");
        prefix_sum_down.compile();

        ocl::Kernel set_0_as_zero(radix_kernel, radix_kernel_length, "set_0_as_zero");
        set_0_as_zero.compile();

        ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        matrix_transpose.compile();

        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
        radix_sort.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            unsigned int workgroup_size = 64;
            unsigned int bit_step = 4;

            unsigned int K = gpu::divup(n, workgroup_size), M = (1 << bit_step);

            gpu::gpu_mem_32u bs_gpu;
            bs_gpu.resizeN(n);

            gpu::gpu_mem_32u cnt_gpu;
            cnt_gpu.resizeN(M * K);

            gpu::gpu_mem_32u cnt_gpu_t;
            cnt_gpu_t.resizeN(M * K);

            gpu::gpu_mem_32u pref_sum;
            pref_sum.resizeN(M * K);
            
            std::vector<unsigned int> cnt(M * K);
            std::vector<unsigned int> cntt(M * K);

            for (unsigned int bit_shift = 0; bit_shift < 32; bit_shift += bit_step) {
                /*as_gpu.readN(as.data(), n);
                for (int i = 0; i < n; ++i)
                    std::cout << as[i] << ' ';
                std::cout << std::endl;
                std::cout << std::endl;*/

                {
                    radix_count.exec(gpu::WorkSize(workgroup_size, gpu::divup(n, workgroup_size) * workgroup_size), 
                        as_gpu, cnt_gpu, bit_shift);
                }

                {
                    unsigned int work_group_size_x = 16;
                    unsigned int work_group_size_y = 16;
                    unsigned int global_work_size_x = gpu::divup(M, work_group_size_x) * work_group_size_x;
                    unsigned int global_work_size_y = gpu::divup(K, work_group_size_y) * work_group_size_y;

                    matrix_transpose.exec(gpu::WorkSize(work_group_size_x, work_group_size_y, global_work_size_x, global_work_size_y), 
                        cnt_gpu, cnt_gpu_t, M, K);
                }
                
                {
                    unsigned int log_n = 32 - __builtin_clz(M * K - 1);
                    for (int d = 0; d < log_n; d++) {
                        prefix_sum_up.exec(gpu::WorkSize(workgroup_size, gpu::divup(M * K / (1 << d), workgroup_size) * workgroup_size),
                            cnt_gpu_t, M * K, d);
                    }
                    set_0_as_zero.exec(gpu::WorkSize(workgroup_size, 1), cnt_gpu_t, M * K);
                    for (int d = log_n - 1; d >= 0; d--) {
                        prefix_sum_down.exec(gpu::WorkSize(workgroup_size, gpu::divup(M * K / (1 << d), workgroup_size) * workgroup_size),
                            cnt_gpu_t, M * K, d);
                    }
                }

                {
                    radix_sort.exec(gpu::WorkSize(workgroup_size, gpu::divup(n, workgroup_size) * workgroup_size), 
                        as_gpu, bs_gpu, cnt_gpu_t, bit_shift, K);
                    bs_gpu.copyToN(as_gpu, n);
                }

                /*as_gpu.readN(as.data(), n);
                cnt_gpu.readN(cnt.data(), M * K);
                cnt_gpu_t.readN(cntt.data(), M * K);
                for (int i = 0; i < n; ++i)
                    std::cout << as[i] << ' ';
                std::cout << std::endl;
                for (int i = 0; i < M * K; ++i)
                    std::cout << cnt[i] << ' ';
                std::cout << std::endl;
                for (int i = 0; i < M * K; ++i)
                    std::cout << cntt[i] << ' ';
                std::cout << std::endl;
                std::cout << std::endl;*/
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
