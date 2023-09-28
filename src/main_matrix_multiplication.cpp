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

struct MultiplicationOnGPU {
    static constexpr int benchmarkingIters = 1;

public:
    MultiplicationOnGPU(uint m, uint k, uint n, const size_t gflops,
                        std::vector<float>&& cpu_as,
                        std::vector<float>&& cpu_bs,
                        std::vector<float>&& cpu_cs)
        : gflops(gflops)
        , M(m), K(k), N(n)
        , as(std::move(cpu_as)), bs(std::move(cpu_bs)), cs(std::move(cpu_cs))
        , cs_cpu_reference(cpu_cs)
    {
        as_gpu.resizeN(m * k);
        bs_gpu.resizeN(k * n);
        cs_gpu.resizeN(m * n);

        as_gpu.writeN(as.data(), m * k);
        bs_gpu.writeN(bs.data(), k * n);
    }

    bool try_multiply(const std::string& name, const std::string& name_func_in_kernel,
                      gpu::WorkSize work_size, std::string defines) {
        std::cout << name << " test:" << std::endl;

        ocl::Kernel kernel(matrix_multiplication,
                           matrix_multiplication_length,
                           name_func_in_kernel,
                           std::move(defines));
        kernel.compile();

        {
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                kernel.exec(work_size,
                            as_gpu, bs_gpu, cs_gpu,
                            M, K, N);

                t.nextLap();
            }
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
            return false;
        }

        std::fill(cs.begin(), cs.end(), 0);

        return true;
    }

private:
    const size_t gflops;
    const uint M;
    const uint K;
    const uint N;

    const std::vector<float> cs_cpu_reference;
    std::vector<float> as;
    std::vector<float> bs;
    std::vector<float> cs;

    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
};

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    constexpr int benchmarkingIters = 10;
    constexpr uint M = 1024;
    constexpr uint K = 1024;
    constexpr uint N = 1024;

    const size_t gflops = ((size_t) M * K * N * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения

    std::vector<float> as(M * K, 0);
    std::vector<float> bs(K * N, 0);
    std::vector<float> cs(M * N, 0);

    FastRandom r(M + K + N);
    for (float& a : as) {
        a = r.nextf();
    }
    for (float& b : bs) {
        b = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N << std::endl;

    {
        timer t;
        for (int iter = 0; iter < 1; ++iter) {
            for (int j = 0; j < M; ++j) {
                for (int i = 0; i < N; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < K; ++k) {
                        sum += as[j * K + k] * bs[k * N + i];
                    }
                    cs[j * N + i] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    constexpr size_t TILE_SIZE = 16;
    constexpr size_t WORK_PER_THREAD = 2;
    std::string defines = "-DTILE_SIZE=" + std::to_string(TILE_SIZE) +
                          " -DWORK_PER_THREAD=" + std::to_string(WORK_PER_THREAD);

    MultiplicationOnGPU multiplicationOnGpu(M, K, N, gflops, std::move(as), std::move(bs), std::move(cs));
    if (!multiplicationOnGpu.try_multiply("Naive", "naive",
                                          gpu::WorkSize{16, 16, N, M},
                                          defines))
        return -1;

    if (!multiplicationOnGpu.try_multiply("With Local Memory", "local_memory",
                                          gpu::WorkSize{16, 16, N, M},
                                          defines))
        return -1;

    if (!multiplicationOnGpu.try_multiply("With Local Memory with more work per thread",
                                          "local_memory_with_more_work_per_thread",
                                          gpu::WorkSize{16, 16 / WORK_PER_THREAD, N, (M + WORK_PER_THREAD - 1) / WORK_PER_THREAD},
                                          defines))
        return -1;

    return 0;
}
