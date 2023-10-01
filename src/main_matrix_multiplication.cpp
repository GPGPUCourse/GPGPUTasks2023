#include <cstdlib>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/matrix_multiplication_cl.h"
#include "libgpu/work_size.h"

#include <iostream>
#include <stdexcept>
#include <vector>

#define TS 16
#define TW 16

gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;

void exec(int benchmarkingIters, int M, int K, int N, std::string kernel_name, gpu::WorkSize ws, const double gflops) {
    ocl::Kernel matrix_multiplication_kernel(matrix_multiplication, matrix_multiplication_length, kernel_name);
    matrix_multiplication_kernel.compile();

    {
        timer t;
        for (int i = 0; i < benchmarkingIters; ++i) {
            matrix_multiplication_kernel.exec(ws, as_gpu, bs_gpu, cs_gpu, M, N, K);
            t.nextLap();
        }

        double avgTime = t.lapAvg();
        std::cout << kernel_name << ": " << avgTime << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << kernel_name << ": " << gflops / avgTime << " GFlops" << std::endl;
    }
}

void check(int M, int K, int N, const std::vector<float> &cs, const std::vector<float> &cs_cpu_reference) {
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
        std::exit(1);
    }
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 1;
    unsigned int M = 1024;
    unsigned int K = 1024;
    unsigned int N = 1024;
    const size_t gflops =
            ((size_t) M * K * N * 2) / (1000 * 1000 * 1000);// умножить на два, т.к. операция сложения и умножения

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

    as_gpu.resizeN(M * K);
    bs_gpu.resizeN(K * N);
    cs_gpu.resizeN(M * N);

    as_gpu.writeN(as.data(), M * K);
    bs_gpu.writeN(bs.data(), K * N);

    exec(benchmarkingIters, M, K, N, "matrix_multiplication_global", gpu::WorkSize(TS, TS, M, N), gflops);
    cs_gpu.readN(cs.data(), M * N);
    check(M, K, N, cs, cs_cpu_reference);

    exec(benchmarkingIters, M, K, N, "matrix_multiplication_local", gpu::WorkSize(TS, TS, M, N), gflops);
    cs_gpu.readN(cs.data(), M * N);
    check(M, K, N, cs, cs_cpu_reference);

    exec(benchmarkingIters, M, K, N, "matrix_multiplication_more_work", gpu::WorkSize(TS, TS / TW, M, N / TW), gflops);
    cs_gpu.readN(cs.data(), M * N);
    check(M, K, N, cs, cs_cpu_reference);
    return 0;
}
