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

    int benchmarkingIters = 100;
    unsigned int n = 1024;
    unsigned int m = 2048;

    std::vector<float> as(n*m, 0);
    std::vector<float> as_t(n*m, 0);

    FastRandom r(n+m);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << n << ", K=" << m << std::endl;

    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(n*m);
    as_t_gpu.resizeN(m*n);

    as_gpu.writeN(as.data(), n*m);

    ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose");
    matrix_transpose_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // TODO
            unsigned int work_group_size = 128;
//            unsigned int global_work_size = ...;
            // Для этой задачи естественнее использовать двухмерный NDRange. Чтобы это сформулировать
            // в терминологии библиотеки - нужно вызвать другую вариацию конструктора WorkSize.
            // В CLion удобно смотреть какие есть вариант аргументов в конструкторах:
            // поставьте каретку редактирования кода внутри скобок конструктора WorkSize -> Ctrl+P -> заметьте что есть 2, 4 и 6 параметров
            // - для 1D, 2D и 3D рабочего пространства соответственно

            matrix_transpose_kernel.exec(gpu::WorkSize(16, 16, m, n), as_gpu, as_t_gpu, n, m);

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << n*m/1000.0/1000.0 / t.lapAvg() << " millions/s" << std::endl;
    }

    as_t_gpu.readN(as_t.data(), n*m);

    // Проверяем корректность результатов
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            float a = as[j * m + i];
            float b = as_t[i * n + j];
            if (a != b) {
                std::cerr << "Not the same!" << std::endl;
                printf("i = %d, j = %d\n", i, j);
                return 1;
            }
        }
    }

    return 0;
}
