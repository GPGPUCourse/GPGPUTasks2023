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

    {
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge");
        merge.compile();
        gpu::gpu_mem_32f as_gpu, bs_gpu;
        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart(); // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных
            unsigned int workGroupSize = 128;
            for (unsigned int block_size = 1; block_size < n; block_size <<= 1)
            {
                merge.exec(gpu::WorkSize(workGroupSize, n), as_gpu, bs_gpu, n, block_size);
                std::swap(as_gpu, bs_gpu);
            }
            t.nextLap();
        }
        as_gpu.readN(as.data(), n);

        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000. / 1000.) / t.lapAvg() << " millions/s" << std::endl;
    }
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    std::cout << "Ok!" << std::endl;
    return 0;
}
