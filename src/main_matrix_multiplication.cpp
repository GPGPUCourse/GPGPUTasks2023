#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10; // TODO пока тестируетесь удобно выставить единицу
    bool skipCPUBenchmarking = true;
    unsigned int M = 1024;
    unsigned int K = 1025;
    unsigned int N = 1023;
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
        for (int iter = 0; iter < benchmarkingIters && (!skipCPUBenchmarking || iter < 1); ++iter) {
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
        std::cout << std::endl;
    }

    auto benchmarkKernel = [&](int kernel_name_id)
    {
        const std::vector<float> cs_cpu_reference = cs;

        gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
        as_gpu.resizeN(M * K);
        bs_gpu.resizeN(K * N);
        cs_gpu.resizeN(M * N);

        as_gpu.writeN(as.data(), M * K);
        bs_gpu.writeN(bs.data(), K * N);

        std::vector<std::string> kernels = {
                "matrix_multiplication_naive0",
                "matrix_multiplication_naive",
                "matrix_multiplication_local_mem_not_coalesced",
                "matrix_multiplication_local_mem_coalesced",
                "matrix_multiplication_more_work_per_thread",
                "matrix_multiplication_local_mem_coalesced2",
                "matrix_multiplication_more_work_per_thread2",
                "matrix_multiplication_more_work_per_thread3"
        };
        std::string kernel_name = kernels[kernel_name_id];
        ocl::Kernel matrix_multiplication_kernel(matrix_multiplication, matrix_multiplication_length,
                                                 kernel_name);
        matrix_multiplication_kernel.compile();

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int work_group_size = 8;
                unsigned int work_per_thread = 8;
                unsigned int global_work_sizeX = (M + work_group_size - 1) / work_group_size * work_group_size;
                unsigned int global_work_sizeY = (N + work_group_size - 1) / work_group_size * work_group_size;

                if (kernel_name_id >= 3)
                    std::swap(global_work_sizeX, global_work_sizeY);
                if (kernel_name_id == 4 || kernel_name_id == 6 || kernel_name_id == 7)
                    global_work_sizeX = (N + work_per_thread * work_group_size - 1) / (work_per_thread * work_group_size) * work_group_size;

                matrix_multiplication_kernel.exec(
                        gpu::WorkSize(work_group_size, work_group_size, global_work_sizeX, global_work_sizeY),
                        as_gpu, bs_gpu, cs_gpu, M, K, N);

                t.nextLap();
            }
            std::cout << "Kernel: " << kernels[kernel_name_id] << std::endl;
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
        }

        cs_gpu.readN(cs.data(), M * N);

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
            exit(1);
        }
        std::cout << std::endl;
    };

    benchmarkKernel(0);
    benchmarkKernel(1);
    benchmarkKernel(2);
    benchmarkKernel(3);
    benchmarkKernel(4);
    benchmarkKernel(5);
    benchmarkKernel(6);
    benchmarkKernel(7);

    return 0;
}
