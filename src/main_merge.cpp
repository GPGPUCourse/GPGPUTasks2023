#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

#include <cassert>
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
    unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        // std::cout << as[i] << ' ';
    }
    // std::cout << std::endl;
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
        std::cout << "CPU: " << (n / 1000.0 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);
    gpu::gpu_mem_32f aux_gpu;
    aux_gpu.resizeN(n);

    // Task 5.1
    auto testGPU = [&](const char *phaseKernelName, gpu::WorkSize workSize) {
        ocl::Kernel mergesortPhase(merge_kernel, merge_kernel_length, phaseKernelName);
        mergesortPhase.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных
            {
                gpu::gpu_mem_32f *from = &as_gpu, *to = &aux_gpu;
                for (int blockLength = 1; blockLength < n; blockLength *= 2) {
                    mergesortPhase.exec(workSize, *from, *to, n, blockLength);
                    std::swap(from, to);
                }
                if (from == &aux_gpu)
                    aux_gpu.copyToN(as_gpu, n);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        std::vector<float> result(n);
        as_gpu.readN(result.data(), n);
        // Проверяем корректность результатов
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(result[i], cpu_sorted[i],
                            "GPU " + std::string(phaseKernelName) + "results should be equal to CPU results!");
        }
    };
    // number of work groups does not really matter
    testGPU("mergesortPhase", gpu::WorkSize(128, n));

    // Task 5.2
    {
        static constexpr int WORK_GROUP_SIZE = 64;
        static constexpr int K = 128;
        ocl::Kernel mergesortPhaseLocal(merge_kernel, merge_kernel_length, "mergesortPhaseLocal");
        mergesortPhaseLocal.compile();
        ocl::Kernel mergesortDiagonalPhase(merge_kernel, merge_kernel_length, "mergesortDiagonalPhase");
        mergesortDiagonalPhase.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных
            {
                gpu::gpu_mem_32f *from = &as_gpu, *to = &aux_gpu;
                for (int blockLength = 1; blockLength < n; blockLength *= 2) {
                    if (blockLength * 2 <= K) {
                        mergesortPhaseLocal.exec(gpu::WorkSize(K, n), *from, *to, n, blockLength);
                    } else {
                        // number of work groups does not really matter
                        mergesortDiagonalPhase.exec(gpu::WorkSize(WORK_GROUP_SIZE, n / K), *from, *to, n, blockLength);
                    }
                    std::swap(from, to);
                    // {
                    //     std::vector<float> current(n);
                    //     from->readN(current.data(), n);
                    //     std::cout << "after blockLength=" << blockLength << ":\n";
                    //     for (float x : current)
                    //         std::cout << x << ' ';
                    //     std::cout << std::endl;
                    // }
                }
                if (from == &aux_gpu)
                    aux_gpu.copyToN(as_gpu, n);
            }
            t.nextLap();
        }
        std::cout << "GPU Diag: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU Diag: " << (n / 1000.0 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);
        // Проверяем корректность результатов
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU Diag results should be equal to CPU results!");
        }
    }


    return 0;
}
