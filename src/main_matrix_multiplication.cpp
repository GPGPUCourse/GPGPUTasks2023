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

    int benchmarkingIters = 10;
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
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    const std::vector<float> cs_cpu_reference = cs;

    
    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(M*K);
    bs_gpu.resizeN(K*N);
    cs_gpu.resizeN(M*N);

    as_gpu.writeN(as.data(), M*K);
    bs_gpu.writeN(bs.data(), K*N);

    ocl::Kernel naive_kernel(matrix_multiplication, matrix_multiplication_length, "naive_matrix_multiplication");
    naive_kernel.compile();

    ocl::Kernel block_kernel(matrix_multiplication, matrix_multiplication_length, "block_matrix_multiplication");
    block_kernel.compile();

    ocl::Kernel threadwork_kernel(matrix_multiplication, matrix_multiplication_length, "threadwork_matrix_multiplication");
    threadwork_kernel.compile();

    // Проверяем корректность результатов
    auto check_result = [&](){
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
    };

    auto run_kernel = [&](const std::string& name, ocl::Kernel& kernel){
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int work_group_size_x = 16;
            unsigned int work_group_size_y = 16;
            unsigned int global_work_size_x = gpu::divup(M, work_group_size_x) * work_group_size_x;
            unsigned int global_work_size_y = gpu::divup(N, work_group_size_y) * work_group_size_y;
            kernel.exec(gpu::WorkSize(work_group_size_x, work_group_size_y, global_work_size_x, global_work_size_y), as_gpu, bs_gpu, cs_gpu, M, K, N);

            t.nextLap();
        }
        std::cout << "GPU (" << name << "): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU (" << name << "): "  << gflops / t.lapAvg() << " GFlops" << std::endl;

        cs_gpu.readN(cs.data(), M*N);
    };

    run_kernel("naive", naive_kernel);
    check_result();

    run_kernel("block", block_kernel);
    check_result();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int work_group_size_x = 1;
            unsigned int work_group_size_y = 16;
            unsigned int global_work_size_x = gpu::divup(gpu::divup(M, 16), work_group_size_x) * work_group_size_x;
            unsigned int global_work_size_y = gpu::divup(N, work_group_size_y) * work_group_size_y;
            threadwork_kernel.exec(gpu::WorkSize(work_group_size_x, work_group_size_y, global_work_size_x, global_work_size_y), as_gpu, bs_gpu, cs_gpu, M, K, N);

            t.nextLap();
        }
        std::cout << "GPU (threadwork): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU (threadwork): "  << gflops / t.lapAvg() << " GFlops" << std::endl;

        cs_gpu.readN(cs.data(), M*N);
    }
    check_result();

    return 0;
}
