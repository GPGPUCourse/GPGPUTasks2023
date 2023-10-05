#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
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

#define WORKGROUP_SIZE 128
#define MAX_WORK_SINGLE 256

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU  : " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU  : " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::gpu_mem_32f as_gpu, bs_gpu;
        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);
        std::vector<float> gpu_sorted(n);
        {
            ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge_one_workgroup");
            merge.compile();
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных
                unsigned int workGroupSize = WORKGROUP_SIZE;
                unsigned int global_work_size = WORKGROUP_SIZE;
                merge.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bs_gpu, n);
                t.nextLap();
            }
            std::cout << "GPU 0: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU 0: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
            as_gpu.readN(gpu_sorted.data(), n);
        }

        // Проверяем корректность результатов
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(gpu_sorted[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
        }
    }

    if (0)
    {
        gpu::gpu_mem_32f as_gpu, bs_gpu;
        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);
        std::vector<float> gpu_sorted(n);
        {
            ocl::Kernel sort_small_blocks(merge_kernel, merge_kernel_length, "sort_small_blocks");
            sort_small_blocks.compile();
            ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge");
            merge.compile();
            ocl::Kernel merge_merge_prepare(merge_kernel, merge_kernel_length, "merge_merge_prepare");
            merge_merge_prepare.compile();
            ocl::Kernel merge_merge(merge_kernel, merge_kernel_length, "merge_merge");
            merge_merge.compile();
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных
                unsigned int workGroupSize = WORKGROUP_SIZE;
                unsigned int global_work_size = ((n + MAX_WORK_SINGLE - 1) / MAX_WORK_SINGLE + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;

                sort_small_blocks.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, n);
                // At this point all blocks of size MAX_WORK_SINGLE = 1 << 8 are sorted

                for (int len = MAX_WORK_SINGLE; len < MAX_WORK_SINGLE * WORKGROUP_SIZE; len <<= 1)
                {
                    merge.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, bs_gpu, len, n);
                    std::swap(as_gpu, bs_gpu);
                }
                // At this point all blocks of size MAX_WORK_SINGLE * WORKGROUP_SIZE = 1 << 15 are sorted

                gpu::gpu_mem_32u ls, rs;
                ls.resizeN(global_work_size / workGroupSize + 1);
                rs.resizeN(global_work_size / workGroupSize + 1);
                for (int len = MAX_WORK_SINGLE * WORKGROUP_SIZE; len < n; len <<= 1)
                {
                    for (int offset = 0; offset + len < n; offset += 2 * len)
                    {
                        merge_merge_prepare.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                                as_gpu, ls, rs,offset, len, n);
                        merge_merge.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                                 as_gpu, bs_gpu, ls, rs,offset, len, n);
                    }
                    //std::swap(as_gpu, bs_gpu);
                }

                t.nextLap();
            }
            std::cout << "GPU 1: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU 1: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
            as_gpu.readN(gpu_sorted.data(), n);
        }

        for (int i = 0; i + 1 < n; i++) {
            if (gpu_sorted[i] > gpu_sorted[i + 1]) {
                std::cout << i << " " << i % (32768) << std::endl;
            }
        }

        // Проверяем корректность результатов
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(gpu_sorted[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
        }
    }

    return 0;
}
