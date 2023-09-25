#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/matrix_multiplication_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>

const int benchmarkingIters = 1;// TODO пока тестируетесь удобно выставить единицу
const unsigned int M = 256;
const unsigned int K = 256;
const unsigned int N = 256;
const double gflops = 2e-9 * M * K * N;// умножить на два, т.к. операция сложения и умножения

gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;

void run_kernel(const char *run_name, const char *kernel_name, const gpu::WorkSize &ws, std::string defines) {
    ocl::Kernel kernel(matrix_multiplication, matrix_multiplication_length, kernel_name, std::move(defines));
    kernel.compile();

    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        kernel.exec(ws, as_gpu, bs_gpu, cs_gpu, M, K, N);
        t.nextLap();
    }
    t.stop();

    double avgTime = t.lapAvg();
    std::cout << run_name << ": " << avgTime << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << run_name << ": " << gflops / avgTime << " GFlops" << std::endl;
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

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

    const size_t TILE_SIZE = 16;
    std::string defines;
    {
        std::ostringstream oss;
        oss << "-DTILE_SIZE=" << TILE_SIZE;
        defines = oss.str();
    }
    run_kernel("GPU Naive", "matrix_multiplication_1", gpu::WorkSize(TILE_SIZE, TILE_SIZE, M, N), defines);
    run_kernel("GPU Local", "matrix_multiplication_2", gpu::WorkSize(TILE_SIZE, TILE_SIZE, M, N), defines);
    run_kernel("GPU Heavy", "matrix_multiplication_3", gpu::WorkSize(TILE_SIZE, TILE_SIZE, M, N), defines);

    cs_gpu.readN(cs.data(), M * N);
    // */

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

    return 0;
}
