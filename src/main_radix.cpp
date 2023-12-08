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
    unsigned int radix_cnt = 4;
    unsigned int wg_size = 128;

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
    gpu::gpu_mem_32u as_gpu, bs_gpu, cs_gpu, ds_gpu, res_gpu;
    int prefix_size = n / wg_size * (1 << radix_cnt);

    as_gpu.resizeN(n);
    bs_gpu.resizeN(prefix_size);
    cs_gpu.resizeN(prefix_size);
    ds_gpu.resizeN(prefix_size);
    res_gpu.resizeN(n);
    std::vector<unsigned int> empty_prefix_vector(prefix_size, 0);
    std::vector<unsigned int> tmp_vector(prefix_size, 0);

    {
        ocl::Kernel radix_count(radix_kernel, radix_kernel_length, "radix_count");
        ocl::Kernel radix_prefix(radix_kernel, radix_kernel_length, "prefix_sum");
        ocl::Kernel radix_prefix_reduce(radix_kernel, radix_kernel_length, "prefix_sum_other");
        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
        radix_count.compile();
        radix_prefix.compile();
        radix_prefix_reduce.compile();
        radix_sort.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();

            for (int radix = 0; radix < 32; radix += radix_cnt) {
                bs_gpu.writeN(empty_prefix_vector.data(), prefix_size);
                cs_gpu.writeN(empty_prefix_vector.data(), prefix_size);
                ds_gpu.writeN(empty_prefix_vector.data(), prefix_size);

                radix_count.exec(gpu::WorkSize(wg_size, n), as_gpu, bs_gpu, n,radix, radix_cnt);


                unsigned int bit_num = 0;
                while (true) {
                    radix_prefix.exec(gpu::WorkSize(wg_size, prefix_size), bs_gpu, ds_gpu, n, bit_num);
                    if ((1 << bit_num) >= prefix_size) {
                        break;
                    }
                    bit_num += 1;
                    unsigned int grid_size = prefix_size / (1 << bit_num);
                    unsigned int wg_size_reduce = std::min(grid_size, wg_size);

                    radix_prefix_reduce.exec(gpu::WorkSize(wg_size_reduce, grid_size), bs_gpu, cs_gpu, n);
                    bs_gpu.swap(cs_gpu);
                }

                radix_sort.exec(gpu::WorkSize(wg_size, n), as_gpu, ds_gpu, res_gpu, n,radix, radix_cnt);
                res_gpu.swap(as_gpu);
            }


            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}