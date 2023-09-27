#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/matrix_transpose_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


void bench_and_check(int benchmarkingIters, uint K, uint M, uint work_group_size, ocl::Kernel kernel,
                     gpu::gpu_mem_32f as_gpu, gpu::gpu_mem_32f as_t_gpu, std::vector<float> const &as,
                     std::string kernel_name) {
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int global_work_size_x = K;
            unsigned int global_work_size_y = M;
            // Для этой задачи естественнее использовать двухмерный NDRange. Чтобы это сформулировать
            // в терминологии библиотеки - нужно вызвать другую вариацию конструктора WorkSize.
            // В CLion удобно смотреть какие есть вариант аргументов в конструкторах:
            // поставьте каретку редактирования кода внутри скобок конструктора WorkSize -> Ctrl+P -> заметьте что есть 2, 4 и 6 параметров
            // - для 1D, 2D и 3D рабочего пространства соответственно
            kernel.exec(gpu::WorkSize(work_group_size, work_group_size, global_work_size_x, global_work_size_y), as_gpu,
                        as_t_gpu, M, K);

            t.nextLap();
        }
        std::cout << "GPU (" << kernel_name << "): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU (" << kernel_name << "): " << M * K / 1000.0 / 1000.0 / t.lapAvg() << " millions/s"
                  << std::endl;
    }

    std::vector<float> as_t(M * K, 0);
    as_t_gpu.readN(as_t.data(), M * K);

   
    // Проверяем корректность результатов
    for (int j = 0; j < M; ++j) {
        for (int i = 0; i < K; ++i) {
            float a = as[j * K + i];
            float b = as_t[i * M + j];
            if (a != b) {
                std::cerr << kernel_name << ": Not the same!" << std::endl;
                return;
            }
        }
    }
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 40;
    unsigned int M = 1024;
    unsigned int K = 1024;

    std::vector<float> as(M * K, 0);

    FastRandom r(M + K);
    for (float &a : as) {
        a = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << std::endl;


    gpu::gpu_mem_32f as_gpu, as_t_gpu;
    as_gpu.resizeN(M * K);
    as_t_gpu.resizeN(K * M);

    as_gpu.writeN(as.data(), M * K);


    {
        int grs = 16;
        ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose",
                                            "-DLOCAL_SIZE=" + std::to_string(grs));
        matrix_transpose_kernel.compile();
        bench_and_check(benchmarkingIters, K, M, grs, matrix_transpose_kernel, as_gpu, as_t_gpu, as,
                        "normal " + std::to_string(grs));
    }


    {
        int grs = 16;
        ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose_bad_banks",
                                            "-DLOCAL_SIZE=" + std::to_string(grs));
        matrix_transpose_kernel.compile();
        bench_and_check(benchmarkingIters, K, M, grs, matrix_transpose_kernel, as_gpu, as_t_gpu, as,
                        "bad_banks " + std::to_string(grs));
    }


    {
        int grs = 16;
        ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose_naive",
                                            "-DLOCAL_SIZE=" + std::to_string(grs));
        matrix_transpose_kernel.compile();
        bench_and_check(benchmarkingIters, K, M, grs, matrix_transpose_kernel, as_gpu, as_t_gpu, as,
                        "naive " + std::to_string(grs));
    }

    {
        int grs = 8;
        ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose",
                                            "-DLOCAL_SIZE=" + std::to_string(grs));
        matrix_transpose_kernel.compile();
        bench_and_check(benchmarkingIters, K, M, grs, matrix_transpose_kernel, as_gpu, as_t_gpu, as,
                        "normal " + std::to_string(grs));
    }

    {
        int grs = 8;
        ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose_bad_banks",
                                            "-DLOCAL_SIZE=" + std::to_string(grs));
        matrix_transpose_kernel.compile();
        bench_and_check(benchmarkingIters, K, M, grs, matrix_transpose_kernel, as_gpu, as_t_gpu, as,
                        "bad_banks " + std::to_string(grs));
    }

    {
        int grs = 8;
        ocl::Kernel matrix_transpose_kernel(matrix_transpose, matrix_transpose_length, "matrix_transpose_naive",
                                            "-DLOCAL_SIZE=" + std::to_string(grs));
        matrix_transpose_kernel.compile();
        bench_and_check(benchmarkingIters, K, M, grs, matrix_transpose_kernel, as_gpu, as_t_gpu, as,
                        "naive " + std::to_string(grs));
    }


    return 0;
}
