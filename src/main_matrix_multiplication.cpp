#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


bool check_reference(std::vector<float> cpu_ref, std::vector<float> gpu_res, unsigned M, unsigned N) {
    double diff_sum = 0;
    for (int i = 0; i < M * N; ++i) {
        double a = gpu_res[i];
        double b = cpu_ref[i];
        if (std::isnan(a) != std::isnan(b) || std::isinf(a) != std::isinf(b)) {
            std::cerr << "Unexpected nan/inf value!" << std::endl;
            return false;
        }
        if (a != 0.0 || b != 0.0) {
            double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
            diff_sum += diff;
        }
    }

    double diff_avg = diff_sum / (M * N);
    std::cout << "Average difference: " << diff_avg * 100.0 << "%" << std::endl;
    if (diff_avg > 0.01) {
        std::cerr << "Too big difference!" << std::endl;
        return false;
    }
    return true;
}


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10; // TODO пока тестируетесь удобно выставить единицу
    unsigned int M = 1024;
    unsigned int K = 1024;
    unsigned int N = 1024;
    const size_t gflops =
            ((size_t) M * K * N * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения

    std::vector<float> as(M * K, 0);
    std::vector<float> bs(K * N, 0);
    std::vector<float> cs(M * N, 0);

    FastRandom r(M + K + N);
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
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    const std::vector<float> cs_cpu_reference = cs;


    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(M * K);
    bs_gpu.resizeN(K * N);
    cs_gpu.resizeN(M * N);

    as_gpu.writeN(as.data(), M * K);
    bs_gpu.writeN(bs.data(), K * N);

    ocl::Kernel matmul_naive_kernel(matrix_multiplication, matrix_multiplication_length, "matmul_naive");
    matmul_naive_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int work_group_size = 16;
            unsigned int grid_size_x = (M + work_group_size - 1) / work_group_size * work_group_size;
            unsigned int grid_size_y = (K + work_group_size - 1) / work_group_size * work_group_size;
            matmul_naive_kernel.exec(gpu::WorkSize(work_group_size, work_group_size, grid_size_x, grid_size_y), as_gpu,
                                     bs_gpu, cs_gpu, M,
                                     K, N);

            t.nextLap();
        }
        std::cout << "GPU naive: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU naive: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    cs_gpu.readN(cs.data(), M * N);



    // Проверяем корректность результатов
    check_reference(cs_cpu_reference, cs, M, N);

    ocl::Kernel matmul_local_kernel(matrix_multiplication, matrix_multiplication_length, "matmul_local");
    matmul_local_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int work_group_size = 16;
            unsigned int grid_size_x = (M + work_group_size - 1) / work_group_size * work_group_size;
            unsigned int grid_size_y = (N + work_group_size - 1) / work_group_size * work_group_size;
            matmul_local_kernel.exec(gpu::WorkSize(work_group_size, work_group_size, grid_size_x, grid_size_y), as_gpu,
                                     bs_gpu, cs_gpu, M,
                                     K, N);

            t.nextLap();
        }
        std::cout << "GPU local: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU local: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    cs_gpu.readN(cs.data(), M * N);


    // Проверяем корректность результатов
    check_reference(cs_cpu_reference, cs, M, N);

    ocl::Kernel matmul_wpt_kernel(matrix_multiplication, matrix_multiplication_length, "matmul_wpt");
    matmul_wpt_kernel.compile();
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // TODO]
            unsigned int wpt = 8;
            unsigned int work_group_size = 16;
            unsigned int grid_size_x = (M + work_group_size - 1) / work_group_size * work_group_size;
            unsigned int grid_size_y = ((K + wpt - 1) / wpt);

            matmul_wpt_kernel.exec(gpu::WorkSize(work_group_size, work_group_size / wpt, grid_size_x,
                                                 grid_size_y),
                                   as_gpu, bs_gpu, cs_gpu, M, K, N);

            t.nextLap();
        }
        std::cout << "GPU wpt: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU wpt: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    cs_gpu.readN(cs.data(), M * N);


    // Проверяем корректность результатов
    check_reference(cs_cpu_reference, cs, M, N);

    return 0;
}
