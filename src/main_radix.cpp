#include "../libs/gpu/libgpu/context.h"
#include "../libs/gpu/libgpu/shared_device_buffer.h"
#include "../libs/utils/libutils/fast_random.h"
#include "../libs/utils/libutils/misc.h"
#include "../libs/utils/libutils/timer.h"

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

#define BITS_AMOUNT 4
#define SIZE_OF_ELEMENT 32
#define WORKGROUP_SIZE 512
#define NUMBERS_AMOUNT (1 << BITS_AMOUNT)


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

    unsigned int workgroup_amount = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    gpu::gpu_mem_32u bs_gpu;
    as_gpu.resizeN(n);

    gpu::gpu_mem_32u counters_gpu;
    as_gpu.resizeN(workgroup_amount * NUMBERS_AMOUNT);

    gpu::gpu_mem_32u prefixes_gpu;
    as_gpu.resizeN(workgroup_amount * NUMBERS_AMOUNT);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        ocl::Kernel bubble_sort(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        ocl::Kernel counters(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        ocl::Kernel prefixes(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            for (int i = 0; SIZE_OF_ELEMENT / BITS_AMOUNT; ++i)
            {

            }
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

    return 0;
}
