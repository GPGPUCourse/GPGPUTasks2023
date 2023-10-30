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

static const unsigned int workGroupSize = 128;

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void radix_sort(std::vector<unsigned int> &as, int n_pow) {
    int n = 1 << n_pow;
    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
    count.compile();
    ocl::Kernel reorder(radix_kernel, radix_kernel_length, "reorder");
    reorder.compile();
    ocl::Kernel prefix_sum_reduce(radix_kernel, radix_kernel_length, "prefix_sum_reduce");
    prefix_sum_reduce.compile();
    ocl::Kernel prefix_sum_write(radix_kernel, radix_kernel_length, "prefix_sum_write");
    prefix_sum_write.compile();
}


void prefix_sum(std::vector<unsigned int> &as, gpu::gpu_mem_32u &dest, int n_pow, ocl::Kernel &prefix_sum_reduce,
                ocl::Kernel &prefix_sum_write) {
    int n = 1 << n_pow;

    gpu::gpu_mem_32u src;
    src.resize(n);
    src.writeN(as.data(), n);

    for (int p = 0; p <= n_pow; ++p) {
        const unsigned int workGroupSize = 128;
        prefix_sum_reduce.exec(gpu::WorkSize(n < workGroupSize ? n : workGroupSize, n), src, p);
        prefix_sum_write.exec(gpu::WorkSize(n < workGroupSize ? n : workGroupSize, n), src, dest, p);
    }
}


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
    /*
    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            // TODO
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
*/
    return 0;
}
