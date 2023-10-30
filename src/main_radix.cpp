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

    const unsigned int work_size = 128;

    int workgroup = n / work_size;
    int total_size = (1 << 4) * workgroup;
    {
        gpu::gpu_mem_32u cnt_gpu;
        cnt_gpu.resizeN(total_size);

        gpu::gpu_mem_32u prefix_gpu;
        prefix_gpu.resizeN(total_size);

        gpu::gpu_mem_32u prefix_swp_gpu;
        prefix_swp_gpu.resizeN(total_size);

        gpu::gpu_mem_32u as_gpu, bs_gpu;
        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);

        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();
        ocl::Kernel radix_cnt(radix_kernel, radix_kernel_length, "radix_cnt");
        radix_cnt.compile();
        ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        matrix_transpose.compile();
        ocl::Kernel prefix(radix_kernel, radix_kernel_length, "prefix");
        prefix.compile();

        timer t;
        for (int i = 0; i < benchmarkingIters; ++i) {
            as_gpu.writeN(as.data(), n);

            t.restart();
            for (unsigned int shift = 0; shift < 32; shift += 4) {

                radix_cnt.exec(gpu::WorkSize(work_size, n), as_gpu, cnt_gpu, shift);

                unsigned int workgroup_size = 16;
                matrix_transpose.exec(gpu::WorkSize(workgroup_size, workgroup_size, 1 << 4, workgroup), cnt_gpu,
                                      prefix_gpu, 1 << 4, workgroup);

                for (int k = 1; k < total_size; k <<= 1) {
                    prefix.exec(gpu::WorkSize(work_size, total_size), prefix_gpu, prefix_swp_gpu, k);
                    prefix_swp_gpu.swap(prefix_gpu);
                }

                radix.exec(gpu::WorkSize(work_size, n), as_gpu, bs_gpu, prefix_gpu, cnt_gpu, shift);
                bs_gpu.swap(as_gpu);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}