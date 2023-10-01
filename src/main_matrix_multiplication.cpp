#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/matrix_multiplication_cl.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <vector>

class MatMulTesting
{
public:
    gpu::Device device;
    gpu::Context context;

    int benchmarkingIters = 10;
    unsigned int M = 1024;
    unsigned int K = 1024;
    unsigned int N = 1024;
    const double gflop =
            ((double) M * K * N * 2) / (1000 * 1000 * 1000);// умножить на два, т.к. операция сложения и умножения

    std::vector<float> as, bs, cs, cs_cpu;

    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;

    MatMulTesting(int argc, char **argv)
        : device(gpu::chooseGPUDevice(argc, argv)), as(M * K, 0), bs(K * N, 0), cs(M * N, 0) {
        context.init(device.device_id_opencl);
        context.activate();
        as_gpu.resizeN(M * K);
        bs_gpu.resizeN(K * N);
        cs_gpu.resizeN(M * N);
    }

    void generateData() {
        FastRandom r(M + K + N);
        for (unsigned int i = 0; i < as.size(); ++i) {
            as[i] = r.nextf();
        }
        for (unsigned int i = 0; i < bs.size(); ++i) {
            bs[i] = r.nextf();
        }
        as_gpu.writeN(as.data(), M * K);
        bs_gpu.writeN(bs.data(), K * N);
        std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N << std::endl;
    }

    void testCPU() {
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
        std::cout << "CPU: " << gflop / t.lapAvg() << " GFlops" << std::endl;
        cs_cpu = cs;
    }

    void testKernel(const char *kernelName, gpu::WorkSize workSize) {
        ocl::Kernel kernel(matrix_multiplication, matrix_multiplication_length, kernelName);
        kernel.compile();
        timer timer;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            kernel.exec(workSize, as_gpu, bs_gpu, cs_gpu, M, K, N);
            timer.nextLap();
        }
        std::cout << "GPU " << kernelName << ": " << timer.lapAvg() << "+-" << timer.lapStd() << std::endl;
        std::cout << "GPU " << kernelName << ": " << gflop / timer.lapAvg() << " GFlops" << std::endl;
        cs_gpu.readN(cs.data(), M * N);
        double diff_sum = 0;
        for (int i = 0; i < M * N; ++i) {
            double a = cs[i];
            double b = cs_cpu[i];
            if (a != 0.0 || b != 0.0) {
                double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
                diff_sum += diff;
            }
        }

        double diff_avg = diff_sum / (M * N);
        std::cout << "Average difference: " << diff_avg * 100.0 << "%" << std::endl;
        if (diff_avg > 0.01)
            throw std::runtime_error("Too big difference!");
    }
};

int main(int argc, char **argv) {
    MatMulTesting testing(argc, argv);
    unsigned M = testing.M;
    unsigned K = testing.K;
    unsigned N = testing.N;
    testing.generateData();

    testing.testCPU();
    testing.testKernel("matrix_multiplication_naive", gpu::WorkSize(32, 1, N, M));
    testing.testKernel("matrix_multiplication_localTile", gpu::WorkSize(32, 32, N, M));
    testing.testKernel("matrix_multiplication_localTileMoreWorkPerThread", gpu::WorkSize(32, 4, N, M / 8));

    return 0;
}
