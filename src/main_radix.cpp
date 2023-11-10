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
#include <cassert>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void print_gpu_mem(const gpu::gpu_mem_32u& as_gpu, unsigned int n) {
    std::vector<unsigned int> res(n, 0);
    as_gpu.readN(res.data(), n);
    std::cout << "\n";
    for (auto a : res)
        std::cout << a << " ";
    std::cout << "\n";
    std::cout << "\n";
}

void print_gpu_mem(const gpu::gpu_mem_32u& as_gpu, unsigned int n, unsigned int m) {
    std::vector<unsigned int> res(n * m, 0);
    as_gpu.readN(res.data(), n * m);

    std::cout << "\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            std::cout << res[i * m + j] << " ";
        std::cout << "\n";
    }
    std::cout << "\n";
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 1;
    unsigned int n = 1024 * 1024 * 32;// * 1024 * 32;// * 1024;

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

    {
        unsigned int wg_merge = 4;
        unsigned int wg_count = 4;
        unsigned int wg_update_blocks = 32;
        unsigned int wg_prefix_sum = 32;
        unsigned int wg_transpose = 32;
        unsigned int tile_size = 4;
        unsigned int wg_sub_prev_row = 32;
        unsigned int k = 2;

        ocl::Kernel merge(radix_kernel, radix_kernel_length, "merge",
                          "-DWG=" + std::to_string(wg_merge) + " "
                                  + "-DK=" + std::to_string(1 << k) + " "
                                  + "-DTILE_SIZE=" + std::to_string(tile_size));
        merge.compile();
        ocl::Kernel count(radix_kernel, radix_kernel_length, "count",
                           "-DWG=" + std::to_string(wg_count) + " "
                                   + "-DK=" + std::to_string(1 << k) + " "
                                   + "-DTILE_SIZE=" + std::to_string(tile_size));
        count.compile();
        ocl::Kernel update_blocks(radix_kernel, radix_kernel_length, "update_blocks",
                           "-DWG=" + std::to_string(wg_update_blocks) + " "
                                   + "-DK=" + std::to_string(1 << k) + " "
                                   + "-DTILE_SIZE=" + std::to_string(tile_size));
        update_blocks.compile();
        ocl::Kernel prefix_sum(radix_kernel, radix_kernel_length, "prefix_sum",
                           "-DWG=" + std::to_string(wg_prefix_sum) + " "
                                   + "-DK=" + std::to_string(1 << k) + " "
                                   + "-DTILE_SIZE=" + std::to_string(tile_size));
        prefix_sum.compile();
        ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose",
                           "-DWG=" + std::to_string(wg_transpose) + " "
                                   + "-DK=" + std::to_string(1 << k) + " "
                                   + "-DTILE_SIZE=" + std::to_string(tile_size));
        matrix_transpose.compile();
        ocl::Kernel sub_prev_row(radix_kernel, radix_kernel_length, "sub_prev_row",
                                     "-DWG=" + std::to_string(wg_sub_prev_row) + " "
                                             + "-DK=" + std::to_string(1 << k) + " "
                                             + "-DTILE_SIZE=" + std::to_string(tile_size));
        matrix_transpose.compile();
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix",
                                 "-DWG=" + std::to_string(wg_count) + " "
                                         + "-DK=" + std::to_string(1 << k) + " "
                                         + "-DTILE_SIZE=" + std::to_string(tile_size));
        radix.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            {
                for(unsigned int offset = 0; offset < sizeof(unsigned int) * 8; offset += k) {
                    {
                        int block_size = 2;
                        for (; block_size <= wg_count; block_size <<= 1) {
                            unsigned int mask = (1 << k) - 1;
                            unsigned int global_work_size = ((n + block_size - 1) / block_size) * block_size;
                            merge.exec(gpu::WorkSize(wg_merge, global_work_size),
                                       as_gpu, bs_gpu, n, block_size, offset, mask);
                            std::swap(as_gpu, bs_gpu);
                        }
                        assert(block_size / 2 == wg_count);
                    }
                    {
                        unsigned int max_elem = (1 << k);
                        unsigned groups_cnt = (n + wg_count - 1) / wg_count;
                        gpu::gpu_mem_32u cnt_gpu;
                        cnt_gpu.resizeN(groups_cnt * max_elem);

                        {
                            unsigned int mask = (1 << k) - 1;
                            unsigned int global_work_size = groups_cnt * wg_count;
                            count.exec(gpu::WorkSize(wg_count, global_work_size),
                                       as_gpu, cnt_gpu, n, offset, mask);
                        }
                        gpu::gpu_mem_32u cnt_pfsum_gpu1;
                        cnt_pfsum_gpu1.resizeN(groups_cnt * max_elem);
                        std::vector<unsigned int> zeros(groups_cnt * max_elem);
                        cnt_pfsum_gpu1.writeN(zeros.data(), groups_cnt * max_elem);
                        gpu::gpu_mem_32u blocks_gpu;
                        blocks_gpu.resizeN(groups_cnt * max_elem);
                        cnt_gpu.copyTo(blocks_gpu, groups_cnt * max_elem * sizeof(unsigned int));
                        {
                            unsigned int N = groups_cnt * max_elem;
                            {
                                unsigned int global_work_size =
                                        (N + wg_prefix_sum - 1) / wg_prefix_sum * wg_prefix_sum;
                                prefix_sum.exec(gpu::WorkSize(wg_prefix_sum, global_work_size),
                                                cnt_pfsum_gpu1, blocks_gpu, 1, N);
                            }
                            for (unsigned int block_size = 2; block_size <= N; block_size <<= 1) {
                                {
                                    unsigned int blocks_count = N / block_size;
                                    unsigned int global_work_size =
                                            ((blocks_count + wg_update_blocks - 1) / wg_update_blocks) *
                                            wg_update_blocks;
                                    update_blocks.exec(gpu::WorkSize(wg_update_blocks, global_work_size),
                                                       blocks_gpu, block_size, N);
                                }
                                {
                                    unsigned int global_work_size =
                                            (N + wg_prefix_sum - 1) / wg_prefix_sum * wg_prefix_sum;
                                    prefix_sum.exec(gpu::WorkSize(wg_prefix_sum, global_work_size),
                                                    cnt_pfsum_gpu1, blocks_gpu, block_size, N);
                                }
                            }
                        }

                        gpu::gpu_mem_32u cnt_t_gpu;
                        cnt_t_gpu.resizeN(groups_cnt * max_elem);
                        {
                            matrix_transpose.exec(gpu::WorkSize(tile_size, tile_size, max_elem, groups_cnt),
                                                  cnt_gpu, cnt_t_gpu, groups_cnt, max_elem);
                        }

                        gpu::gpu_mem_32u cnt_t_pfsum_gpu;
                        cnt_t_pfsum_gpu.resizeN(groups_cnt * max_elem);
                        cnt_t_pfsum_gpu.writeN(zeros.data(), groups_cnt * max_elem);
                        cnt_t_gpu.copyTo(blocks_gpu, groups_cnt * max_elem * sizeof(unsigned int));
                        {
                            unsigned int N = groups_cnt * max_elem;
                            {
                                unsigned int global_work_size =
                                        (N + wg_prefix_sum - 1) / wg_prefix_sum * wg_prefix_sum;
                                prefix_sum.exec(gpu::WorkSize(wg_prefix_sum, global_work_size),
                                                cnt_t_pfsum_gpu, blocks_gpu, 1, N);
                            }
                            for (unsigned int block_size = 2; block_size <= N; block_size <<= 1) {
                                {
                                    unsigned int blocks_count = N / block_size;
                                    unsigned int global_work_size =
                                            ((blocks_count + wg_update_blocks - 1) / wg_update_blocks) *
                                            wg_update_blocks;
                                    update_blocks.exec(gpu::WorkSize(wg_update_blocks, global_work_size),
                                                       blocks_gpu, block_size, N);
                                }
                                {
                                    unsigned int global_work_size =
                                            (N + wg_prefix_sum - 1) / wg_prefix_sum * wg_prefix_sum;
                                    prefix_sum.exec(gpu::WorkSize(wg_prefix_sum, global_work_size),
                                                    cnt_t_pfsum_gpu, blocks_gpu, block_size, N);
                                }
                            }
                        }

                        gpu::gpu_mem_32u cnt_pfsum_gpu2;
                        cnt_pfsum_gpu2.resizeN(groups_cnt * max_elem);
                        {
                            unsigned int global_work_size = (max_elem * groups_cnt + wg_sub_prev_row - 1) / wg_sub_prev_row * wg_sub_prev_row;
                            sub_prev_row.exec(gpu::WorkSize(wg_sub_prev_row, global_work_size),
                                              cnt_pfsum_gpu1, cnt_pfsum_gpu2, groups_cnt, max_elem);
                        }
                        {
                            unsigned int mask = (1 << k) - 1;
                            unsigned int global_work_size = groups_cnt * wg_count;
                            radix.exec(gpu::WorkSize(wg_count, global_work_size),
                                       as_gpu, bs_gpu, cnt_t_pfsum_gpu, cnt_pfsum_gpu2, n, offset, mask);
                        }
                        std::swap(as_gpu, bs_gpu);
                    }
                }
                t.nextLap();
            }
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
