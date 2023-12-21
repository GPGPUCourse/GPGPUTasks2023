#include "../libs/gpu/libgpu/context.h"
#include "../libs/gpu/libgpu/shared_device_buffer.h"
#include "../libs/utils/libutils/fast_random.h"
#include "../libs/utils/libutils/misc.h"
#include "../libs/utils/libutils/timer.h"
#include "../libs/gpu/libgpu/work_size.h"

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

#define WORKGROUP_SIZE 128

unsigned int closedPowerOfTwo(unsigned int n) {
    unsigned int result = 1;
    while (result < n) {
        result *= 2;
    }
    return result;
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32;
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
    unsigned int workn = closedPowerOfTwo(n);
    std::cout << "workn = " << workn << std::endl;
    as_gpu.resizeN(workn);

    auto as_complete = std::vector<float>(workn);

    for (int i = 0; i < workn; ++i) {
        if (i < as.size())
        {
            as_complete[i] = as[i];
        }
        else
        {
            as_complete[i] = std::numeric_limits<float>::infinity();
        }
    }
    for (int i = 0; i < as.size(); ++i) {
        std::cout << as[i] << " ";
    }
    std::cout << std::endl;

    for (int i = 0; i < as_complete.size(); ++i) {
        std::cout << as_complete[i] << " ";
    }
    std::cout << std::endl;

    {
        std::cout << "GPU: " << std::endl;
        ocl::Kernel bitonic(bitonic_kernel, bitonic_kernel_length, "bitonic");
        std::cout << "Compiling" << std::endl;
        bitonic.compile();

        std::cout << "Compiled" << std::endl;
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as_complete.data(), workn);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            std::cout << "Started" << std::endl;
            unsigned int block_size = 2;
            while (block_size <= n) {
                bitonic.exec(gpu::WorkSize(WORKGROUP_SIZE, workn), as_gpu, workn, block_size);
                std::cout << "block_size = " << block_size << std::endl;
                block_size *= 2;
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
