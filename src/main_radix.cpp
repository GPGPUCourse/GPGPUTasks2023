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
#define WORKGROUP_SIZE 128
#define NUMBERS_AMOUNT 16
#define WORKGROUP_SIZE_2D 16

gpu::WorkSize work_size(int size) {
    unsigned int global_size = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;
    return gpu::WorkSize(WORKGROUP_SIZE, global_size);
}

gpu::WorkSize work_size2d(int x, int y) {
    unsigned int global_size_x = (x + WORKGROUP_SIZE_2D - 1) / WORKGROUP_SIZE_2D * WORKGROUP_SIZE_2D;
    unsigned int global_size_y = (y + WORKGROUP_SIZE_2D - 1) / WORKGROUP_SIZE_2D * WORKGROUP_SIZE_2D;
    return gpu::WorkSize(WORKGROUP_SIZE_2D, WORKGROUP_SIZE_2D, global_size_x, global_size_y);
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

//    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    int benchmarkingIters = 1;
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

    std::cout << std::endl;
    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(n);

    gpu::gpu_mem_32u cs_gpu;
    cs_gpu.resizeN(n);
    std::vector<unsigned int> cs(n);

    gpu::gpu_mem_32u counters_gpu;
    counters_gpu.resizeN(workgroup_amount * NUMBERS_AMOUNT);

    gpu::gpu_mem_32u counters_transposed_gpu;
    counters_transposed_gpu.resizeN(workgroup_amount * NUMBERS_AMOUNT);

    gpu::gpu_mem_32u prefixes_gpu;
    prefixes_gpu.resizeN(workgroup_amount * NUMBERS_AMOUNT);

    gpu::gpu_mem_32u prefixes_buffer_gpu;
    prefixes_buffer_gpu.resizeN(workgroup_amount * NUMBERS_AMOUNT);

    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        ocl::Kernel bubble_sort(radix_kernel, radix_kernel_length, "small_merge_sort");
        bubble_sort.compile();

        ocl::Kernel counters(radix_kernel, radix_kernel_length, "counters");
        counters.compile();

        ocl::Kernel prefixes(radix_kernel, radix_kernel_length, "prefixes");
        prefixes.compile();

        ocl::Kernel matrix_transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        matrix_transpose.compile();

        ocl::Kernel copy(radix_kernel, radix_kernel_length, "copy");
        copy.compile();

        ocl::Kernel zero(radix_kernel, radix_kernel_length, "zero");
        zero.compile();
        zero.exec(work_size(NUMBERS_AMOUNT * workgroup_amount), prefixes_gpu, NUMBERS_AMOUNT * workgroup_amount);
        zero.exec(work_size(NUMBERS_AMOUNT * workgroup_amount), counters_gpu, NUMBERS_AMOUNT * workgroup_amount);
        zero.exec(work_size(NUMBERS_AMOUNT * workgroup_amount), counters_transposed_gpu, NUMBERS_AMOUNT * workgroup_amount);
        unsigned int numbers_amount = NUMBERS_AMOUNT;
        std::cout << "Workgroups = " << workgroup_amount << std::endl;
        std::cout << "Global worksize = " << ((n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE) << std::endl;
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            for (int i = 0; i < SIZE_OF_ELEMENT / BITS_AMOUNT; ++i)
            {
                bubble_sort.exec(work_size(n), as_gpu, cs_gpu, i * BITS_AMOUNT);
                std::swap(as_gpu, cs_gpu);
                counters.exec(work_size(n), as_gpu, counters_gpu, i * BITS_AMOUNT, n, workgroup_amount);
                matrix_transpose.exec(work_size2d(NUMBERS_AMOUNT, workgroup_amount), counters_gpu, counters_transposed_gpu, workgroup_amount, numbers_amount);
                copy.exec(work_size(workgroup_amount * NUMBERS_AMOUNT), counters_transposed_gpu, prefixes_buffer_gpu, workgroup_amount * NUMBERS_AMOUNT);
                for (int j = 1; (1 << (j-1)) < workgroup_amount * NUMBERS_AMOUNT; ++j)
                {
                    prefixes.exec(work_size(workgroup_amount * NUMBERS_AMOUNT), prefixes_buffer_gpu, prefixes_gpu, workgroup_amount * NUMBERS_AMOUNT, j);
                    std::swap(prefixes_buffer_gpu, prefixes_gpu);
                }
                std::swap(prefixes_buffer_gpu, prefixes_gpu);
                radix.exec(work_size(n), as_gpu, counters_transposed_gpu, prefixes_gpu, bs_gpu, i * BITS_AMOUNT, n, workgroup_amount);
                zero.exec(work_size(NUMBERS_AMOUNT * workgroup_amount), prefixes_gpu, NUMBERS_AMOUNT * workgroup_amount);
                zero.exec(work_size(NUMBERS_AMOUNT * workgroup_amount), counters_gpu, NUMBERS_AMOUNT * workgroup_amount);
                std::swap(as_gpu, bs_gpu);
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
