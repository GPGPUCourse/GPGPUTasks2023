#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_transpose_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    uint M = 1024;
    uint K = 1024;
    uint n = M * K;

    std::vector<float> as(n, 0);
    std::vector<float> as_t(n, 0);

    FastRandom r(n);
    for (float& a : as) {
        a = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << std::endl;

    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(n);
    as_t_gpu.resizeN(n);

    as_gpu.writeN(as.data(), n);

    ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose");
    matrix_transpose_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            uint work_group_size = 16;

            if (M % work_group_size != 0 || K % work_group_size != 0)
                throw std::runtime_error("");

            uint global_work_size_X = K;
            uint global_work_size_Y = M;
            // Для этой задачи естественнее использовать двухмерный NDRange. Чтобы это сформулировать
            // в терминологии библиотеки - нужно вызвать другую вариацию конструктора WorkSize.
            gpu::WorkSize work_size(work_group_size, work_group_size, global_work_size_X, global_work_size_Y);
            matrix_transpose_kernel.exec(work_size, as_gpu, as_t_gpu, M, K);

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << M*K/1000.0/1000.0 / t.lapAvg() << " millions/s" << std::endl;
    }

    as_t_gpu.readN(as_t.data(), M * K);

    // Проверяем корректность результатов
    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < K; ++i) {
            float a = as[j * K + i];
            float b = as_t[i * M + j];
            if (a != b) {
                std::cerr << "Not the same!" << std::endl;
                return 1;
            }
        }
    }

    return 0;
}
