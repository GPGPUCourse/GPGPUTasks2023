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
    unsigned int n = 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    std::vector<unsigned int> debug(n, 0);
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

    const unsigned int digits_number = 4;
    const unsigned int bits_number = 2;
    const unsigned int bits_in_uint = sizeof(unsigned int) * 8;
    std::cout << "bits_in_uint: "<< bits_in_uint << '\n';
    const unsigned int work_group_size = 32;
    const unsigned int global_work_size = n;
    const unsigned int work_groups_number = n / work_group_size + 1;
    const unsigned int counters_size = work_groups_number * digits_number;

    const std::vector<unsigned int> zeros(n, 0);
    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);
    gpu::gpu_mem_32u as_sorted_gpu;
    as_sorted_gpu.resizeN(n);
    gpu::gpu_mem_32u counters_gpu;
    counters_gpu.resizeN(counters_size);
    gpu::gpu_mem_32u prefix_sums_gpu;
    prefix_sums_gpu.resizeN(counters_size);

    {
        ocl::Kernel radix_count(radix_kernel, radix_kernel_length, "radix_count");
        radix_count.compile();

        ocl::Kernel radix_prefix_sum(radix_kernel, radix_kernel_length, "radix_prefix_sum");
        radix_prefix_sum.compile();

        ocl::Kernel radix_prefix_sum_reduce(radix_kernel, radix_kernel_length, "radix_prefix_sum_reduce");
        radix_prefix_sum_reduce.compile();

        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
        radix_sort.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();
            for (int bits_offset = 0; bits_offset <= bits_in_uint; bits_offset += bits_number) {
                counters_gpu.writeN(zeros.data(), counters_size);
                prefix_sums_gpu.writeN(zeros.data(), counters_size);
                // as_sorted_gpu.writeN(zeros.data(), n);
                std::cout << "DEBUG 1\n";
                radix_count.exec(gpu::WorkSize(work_group_size, global_work_size), as_gpu, n, counters_gpu,
                                 work_groups_number, bits_offset);
                for (unsigned int cur_block_size = 1; cur_block_size <= work_groups_number; cur_block_size <<= 1) {
                    radix_prefix_sum.exec(gpu::WorkSize(work_group_size, work_groups_number), counters_gpu,
                                          prefix_sums_gpu, work_groups_number, cur_block_size);
                    radix_prefix_sum_reduce.exec(gpu::WorkSize(work_group_size, work_groups_number / cur_block_size),
                                                 counters_gpu, work_groups_number, cur_block_size);
                }
                radix_sort.exec(gpu::WorkSize(work_group_size, global_work_size), as_gpu, as_sorted_gpu, n,
                                prefix_sums_gpu, work_groups_number, bits_offset);
                as_sorted_gpu.swap(as_gpu);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_sorted_gpu.readN(as.data(), n);
    }

    for (int i = 0; i < n; ++i) {
        if (as[i] != cpu_sorted[i]) {
            std::cout << as[i] << std::endl;
            for (int w = 0; w < 32; w++) {
                std::cout << as[i + w] << " ";
            }
            std::cout << "\n";
            for (int w = 0; w < 32; w++) {
                std::cout << cpu_sorted[i + w] << " ";
            }
            std::cout << "\n";
        }
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
