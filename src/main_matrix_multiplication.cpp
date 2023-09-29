#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>

int checkMultiplication(const gpu::gpu_mem_32f& as_gpu, const gpu::gpu_mem_32f& bs_gpu, unsigned int N, unsigned int K, unsigned int M,
                        const std::vector<float>& cs_cpu_reference,
                        const std::string& kernelName,
                        gpu::WorkSize workSize,
                        int benchmarkingIters = 10) {
    const size_t gflops = ((size_t) M * K * N * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения
    std::vector<float> cs(M*N, 0);
    gpu::gpu_mem_32f cs_gpu;
    cs_gpu.resizeN(M*N);

    ocl::Kernel matrix_multiplication_kernel(matrix_multiplication, matrix_multiplication_length, kernelName);
    matrix_multiplication_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            matrix_multiplication_kernel.exec(workSize, as_gpu, bs_gpu, cs_gpu, M, K, N);
            t.nextLap();
        }
        std::cout << "GPU: " << kernelName << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << kernelName << ": " << gflops / t.lapAvg() << " GFlops" << std::endl;
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

    return 0;
}

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

    gpu::gpu_mem_32f as_gpu, bs_gpu;
    as_gpu.resizeN(M*K);
    bs_gpu.resizeN(K*N);

    as_gpu.writeN(as.data(), M*K);
    bs_gpu.writeN(bs.data(), K*N);

    const unsigned int groupSize = 16;

    unsigned int workSizeX = (N + groupSize - 1) / groupSize * groupSize;
    unsigned int workSizeY = (M + groupSize - 1) / groupSize * groupSize;

    // 4.2.0 Реализуйте наивное умножение матриц через глобальную память
    if (int ret = checkMultiplication(as_gpu, bs_gpu, N, K, M,
                                  cs_cpu_reference,
                                  "matrix_multiplication_naive",
                                  gpu::WorkSize(groupSize, groupSize, workSizeX, workSizeY),
                                  benchmarkingIters)) {
        return ret;
    }

    // 4.2.1 Реализуйте умножение матриц через локальную память. (на лекции это вплоть до "Умножение матриц 2: локальная память")
    if (int ret = checkMultiplication(as_gpu, bs_gpu, N, K, M,
                                      cs_cpu_reference,
                                      "matrix_multiplication_local",
                                      gpu::WorkSize(groupSize, groupSize, workSizeX, workSizeY),
                                      benchmarkingIters)) {
        return ret;
    }

    // 4.2.2 Реализуйте умножение матриц через локальную память с большим количеством работы
    // на один воркайтем (на лекции на странице "Умножение матриц 3: more work per thread".
    // На странице с дожиманиями перемножения матриц это кернел №3).
    // Обратите внимание на то чтобы после перехода на эту версию не потерять существенную
    // часть коалесд-доступа к глобальной памяти. Этого удобно добиться пойдя от обратного,
    // начав с того, какие клеточки хочется чтобы варп записал соседними потоками,
    // и выбрав индексацию так, чтобы выбранному порядку соответствовать.

    const unsigned int threadWorkGroupSize = 16;
    const unsigned int threadWork = 4;

    unsigned int workSizeThreadWorkX = (N + threadWorkGroupSize - 1) / threadWorkGroupSize * threadWorkGroupSize;
    unsigned int workSizeThreadWorkY = (M + threadWorkGroupSize * threadWork - 1) / threadWorkGroupSize / threadWork * threadWorkGroupSize;

    if (int ret = checkMultiplication(as_gpu, bs_gpu, N, K, M,
                                      cs_cpu_reference,
                                      "matrix_multiplication_more_work_per_thread",
                                      gpu::WorkSize(threadWorkGroupSize, threadWorkGroupSize / threadWork,
                                                    workSizeThreadWorkX, workSizeThreadWorkY),
                                      benchmarkingIters)) {
        return ret;
    }

    return 0;
}
