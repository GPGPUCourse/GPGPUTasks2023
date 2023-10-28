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
    std::vector<unsigned int> zeroes(n, 0);
    gpu::gpu_mem_32u as_gpu, tmp_gpu, cs_gpu, cst_gpu, ps_gpu, psb_gpu;
    as_gpu.resizeN(n);
    tmp_gpu.resizeN(n);

    {
        unsigned int work_group_size = 256;
        unsigned int tile_size = 64;
        unsigned int value_bytes = 2;
        unsigned int keys = 1 << value_bytes;
        unsigned int c_size = n / work_group_size * keys;

        cs_gpu.resizeN(c_size);
        cst_gpu.resizeN(c_size);
        ps_gpu.resizeN(c_size);

        ocl::Kernel merge(radix_kernel, radix_kernel_length, "merge_small");
        merge.compile();
        auto merge_work_size = gpu::WorkSize(work_group_size, n);
        ocl::Kernel get_counts(radix_kernel, radix_kernel_length, "get_counts");
        get_counts.compile();
        auto get_counts_work_size = gpu::WorkSize(work_group_size, n);
        ocl::Kernel zero(radix_kernel, radix_kernel_length, "zero");
        zero.compile();
        auto zero_work_size = gpu::WorkSize(work_group_size, c_size);
        ocl::Kernel reduce(radix_kernel, radix_kernel_length, "reduce");
        reduce.compile();
        ocl::Kernel add_to_result(radix_kernel, radix_kernel_length, "add_to_result");
        add_to_result.compile();
        auto add_to_result_work_size = gpu::WorkSize(work_group_size, c_size / 2);
        ocl::Kernel transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        transpose.compile();
        auto transpose_work_size = gpu::WorkSize(keys, tile_size, keys, n / work_group_size);
        ocl::Kernel local_prefix(radix_kernel, radix_kernel_length, "local_prefix");
        local_prefix.compile();
        auto local_prefix_work_size = gpu::WorkSize(work_group_size, n / work_group_size);
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();
        auto radix_work_size = gpu::WorkSize(work_group_size, n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            cs_gpu.writeN(zeroes.data(), c_size);
            ps_gpu.writeN(zeroes.data(), c_size);
            /*
            for (auto v: as) {
                std::cout << v << " ";
            }
            std::cout << std::endl;
             */

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            for (unsigned int stage=0; stage<32/value_bytes; ++stage) {
                for (unsigned int block_size=1; block_size<work_group_size; block_size*=2) {
                    merge.exec(merge_work_size, as_gpu, tmp_gpu, block_size, stage);
                    std::swap(as_gpu, tmp_gpu);
                }

                /*
                as_gpu.readN(as.data(), n / work_group_size * keys);
                for (auto v: as) {
                    std::cout << v << " ";
                }
                std::cout << std::endl;
                 */

                get_counts.exec(get_counts_work_size, as_gpu, cs_gpu, stage);

                /*
                cs_gpu.readN(as.data(), c_size);
                for (auto v: as) {
                    std::cout << v << " ";
                }
                std::cout << std::endl;
                 */

                transpose.exec(transpose_work_size, cs_gpu, cst_gpu, n / work_group_size);

                /*
                cst_gpu.readN(as.data(), c_size);
                for (auto v: as) {
                    std::cout << v << " ";
                }
                std::cout << std::endl;
                 */

                psb_gpu = cs_gpu;
                local_prefix.exec(local_prefix_work_size, cst_gpu, psb_gpu);

                /*
                psb_gpu.readN(as.data(), c_size);
                for (auto v: as) {
                    std::cout << v << " ";
                }
                std::cout << std::endl;
                 */

                zero.exec(zero_work_size, ps_gpu);
                for (unsigned int block_size=1; block_size < c_size; block_size *= 2) {
                    add_to_result.exec(add_to_result_work_size, cst_gpu, ps_gpu, block_size);
                    auto reduce_work_size = gpu::WorkSize(work_group_size, c_size / (2 * block_size));
                    reduce.exec(reduce_work_size, cst_gpu, block_size, c_size);
                }

                /*
                ps_gpu.readN(as.data(), c_size);
                for (auto v: as) {
                    std::cout << v << " ";
                }
                std::cout << std::endl;
                 */

                radix.exec(radix_work_size, as_gpu, ps_gpu, psb_gpu, tmp_gpu, stage);
                std::swap(as_gpu, tmp_gpu);

                /*
                as_gpu.readN(as.data(), n);
                for (auto v: as) {
                    std::cout << v << " ";
                }
                std::cout << std::endl;
                */

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
