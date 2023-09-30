#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <stdexcept>

#define THREAD_WORK 32
#define TILE_SIZE 32

const int benchmarkingIters = 1; // TODO пока тестируетесь удобно выставить единицу
const unsigned int M = 1024;
const unsigned int K = 1024;
const unsigned int N = 1024;
const size_t gflops = ((size_t) M * K * N * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения


void run_kernel(std::vector<float> &as,
                std::vector<float> &bs,
                std::vector<float> &cs,
                const std::vector<float> &cs_cpu_reference,
                std::string kernel_name,
                int work_group_size_x,
                int work_group_size_y,
                int global_work_size_x,
                int global_work_size_y) {


    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(M*K);
    bs_gpu.resizeN(K*N);
    cs_gpu.resizeN(M*N);

    as_gpu.writeN(as.data(), M*K);
    bs_gpu.writeN(bs.data(), K*N);
    std::fill(cs.begin(), cs.end(), 0);

    ocl::Kernel matrix_multiplication_kernel(matrix_multiplication, matrix_multiplication_length, kernel_name,
                    "-D TILE_SIZE="+std::to_string(work_group_size_x) +
                    " -D THREAD_WORK="+std::to_string(THREAD_WORK) +
                    " -D GROUP_WORK="+std::to_string(work_group_size_x / THREAD_WORK));
    matrix_multiplication_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            matrix_multiplication_kernel.exec(gpu::WorkSize(work_group_size_x, work_group_size_y, global_work_size_x, global_work_size_y), as_gpu, bs_gpu, cs_gpu, M, K, N);

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
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
    }
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();
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

    run_kernel(as, bs, cs, cs_cpu_reference, "matrix_multiplication_1", TILE_SIZE, TILE_SIZE, M, N);
    run_kernel(as, bs, cs, cs_cpu_reference, "matrix_multiplication_2", TILE_SIZE, TILE_SIZE, M, N);
    run_kernel(as, bs, cs, cs_cpu_reference, "matrix_multiplication_3", TILE_SIZE, TILE_SIZE / THREAD_WORK, M, N / THREAD_WORK);
    return 0;
}
