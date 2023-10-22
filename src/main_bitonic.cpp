#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"

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
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }
    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);

    {
        ocl::Kernel bitonic(bitonic_kernel, bitonic_kernel_length, "bitonic");
        bitonic.compile();
        ocl::Kernel bitonic_local(bitonic_kernel, bitonic_kernel_length, "bitonic_local");
        bitonic_local.compile();

        timer t;
        unsigned int wg_size = 128;
        unsigned int grid_size = (n + wg_size - 1) / wg_size * wg_size;

        // Ищем наименьшую степень двойки, большую нашего n
        unsigned int max_size = 2;
        while (max_size < n) max_size <<= 1;

        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();

            // Начинаем с сортировки подмассивов длины 2
            unsigned int size = 2;

            while (size <= max_size) {
                // Размер части массива одного "тона"
                unsigned int batch_size = size >> 1;

                // Обработка больших кусков
                while (2 * batch_size > wg_size) {
                    bitonic.exec(gpu::WorkSize(wg_size, grid_size), as_gpu, batch_size, size, n);
                    batch_size >>= 1;
                }

                // Локальная обработка
                bitonic_local.exec(gpu::WorkSize(wg_size, grid_size), as_gpu, batch_size, size, n);
                size <<= 1;
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
