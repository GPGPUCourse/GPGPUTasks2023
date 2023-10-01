#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <utility>
#include <vector>
#include <iostream>
#include <stdexcept>


int test_kernel(std::string kernel_name, gpu::WorkSize work_size,
                const gpu::gpu_mem_32f& as_gpu,
                const gpu::gpu_mem_32f& bs_gpu,
                const gpu::gpu_mem_32f& cs_gpu,
                unsigned int M, unsigned int K, unsigned int N,
                std::vector<float> cs,
                std::vector<float> cs_cpu_reference,
                int benchmarkingIters) {
    std::cout << kernel_name << " kernel test:" << std::endl;
    const size_t gflops = ((size_t) M * K * N * 2) / (1000 * 1000 * 1000);

    ocl::Kernel matrix_multiplication_kernel(matrix_multiplication, matrix_multiplication_length, std::move(kernel_name));
    matrix_multiplication_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            matrix_multiplication_kernel.exec(work_size, as_gpu, bs_gpu, cs_gpu, M, K, N);

            t.nextLap();
        }
        std::cout << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    cs_gpu.readN(cs.data(), M*N);

    // Проверяем корректность результатов
    double diff_sum = 0;
    for (int i = 0; i < M * N; ++i) {
        double a = cs[i];
        double b = cs_cpu_reference[i];
        if (a != 0.0 || b != 0.0) {
            double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
            diff_sum += diff;
        }
    }

    double diff_avg = diff_sum / (M * N);
    std::cout << "Average difference: " << diff_avg * 100.0 << "%" << std::endl;
    if (diff_avg > 0.01) {
        std::cerr << "Too big difference!" << std::endl;
        return 1;
    }
    std::cout << std::endl;
    return 0;
}


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10; // TODO пока тестируетесь удобно выставить единицу
    unsigned int M = 1024;
    unsigned int K = 1024;
    unsigned int N = 1024;
    const size_t gflops = ((size_t) M * K * N * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения

    std::vector<float> as(M*K, 0);
    std::vector<float> bs(K*N, 0);
    std::vector<float> cs(M*N, 0);

    FastRandom r(M+K+N);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    for (unsigned int i = 0; i < bs.size(); ++i) {
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N << std::endl;

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += as.data()[j * K + k] * bs.data()[k * N + i];
                    }
                    cs.data()[j * N + i] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU test:" << std::endl;
        std::cout << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << gflops / t.lapAvg() << " GFlops" << std::endl;
        std::cout << std::endl;
    }

    const std::vector<float> cs_cpu_reference = cs;

    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(M*K);
    bs_gpu.resizeN(K*N);
    cs_gpu.resizeN(M*N);

    as_gpu.writeN(as.data(), M*K);
    bs_gpu.writeN(bs.data(), K*N);

    {
        unsigned int work_group_size = 16;
        unsigned int global_work_size_x = (N + work_group_size - 1) / work_group_size * work_group_size;
        unsigned int global_work_size_y = (M + work_group_size - 1) / work_group_size * work_group_size;
        gpu::WorkSize work_size = gpu::WorkSize(work_group_size, work_group_size, global_work_size_x, global_work_size_y);
        int test_result = test_kernel(
            "matrix_multiplication_base",
            work_size,
            as_gpu,
            bs_gpu,
            cs_gpu,
            M, K, N,
            cs, cs_cpu_reference,
            benchmarkingIters * 10
        );
        if (test_result != 0) {
            return 1;
        }
    }

    {
        unsigned int tile_size = 16;
        unsigned int global_work_size_x = (N + tile_size - 1) / tile_size * tile_size;
        unsigned int global_work_size_y = (M + tile_size - 1) / tile_size * tile_size;
        gpu::WorkSize work_size = gpu::WorkSize(tile_size, tile_size, global_work_size_x, global_work_size_y);
        int test_result = test_kernel(
                "matrix_multiplication_local_mem",
                work_size,
                as_gpu,
                bs_gpu,
                cs_gpu,
                M, K, N,
                cs, cs_cpu_reference,
                benchmarkingIters
        );
        if (test_result != 0) {
            return 1;
        }
    }

    {
        unsigned int tile_size = 16;
        unsigned int values_per_work_item = 8;
        unsigned int global_work_size_x = (N + tile_size - 1) / tile_size * tile_size;
        unsigned int global_work_size_y = ((M + values_per_work_item - 1) / values_per_work_item + tile_size - 1) / tile_size * tile_size;
        gpu::WorkSize work_size = gpu::WorkSize(tile_size, tile_size, global_work_size_x, global_work_size_y);
        int test_result = test_kernel(
                "matrix_multiplication_busy",
                work_size,
                as_gpu,
                bs_gpu,
                cs_gpu,
                M, K, N,
                cs, cs_cpu_reference,
                benchmarkingIters
        );
        if (test_result != 0) {
            return 1;
        }
    }


    return 0;
}
