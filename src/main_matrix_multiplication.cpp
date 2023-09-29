#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/matrix_multiplication_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>

int check_correct(const std::vector<float> &cs_cpu, const std::vector<float> &cs, unsigned int N, unsigned int M) {
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
    if (diff_avg > 0.01) {
        std::cerr << "Too big difference!" << std::endl;
        return 1;
    }
    return 0;
}

template<typename F>
void helper(ocl::Kernel &kernel, const F &f, const std::string& prefix, const unsigned int benchmarkingIters, const unsigned int gflops) {
    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        f(kernel);
        t.nextLap();
    }
    std::cout << std::endl << prefix << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << prefix << gflops / t.lapAvg() << " GFlops" << std::endl;
}

template<typename F>
void executor(const std::string &func, const std::string &prefix, const F &f, const unsigned int benchmarkingIters, const unsigned int gflops) {
    ocl::Kernel kernel(matrix_multiplication, matrix_multiplication_length, func);
    kernel.compile();
    helper(kernel, f, prefix, benchmarkingIters, gflops);
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;// TODO пока тестируетесь удобно выставить единицу
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

    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(M * K);
    bs_gpu.resizeN(K * N);
    cs_gpu.resizeN(M * N);

    as_gpu.writeN(as.data(), M * K);
    bs_gpu.writeN(bs.data(), K * N);

    std::cout << std::endl << "selected device: " << device.name << std::endl;

    {
        auto func = [&](ocl::Kernel &k) { k.exec(gpu::WorkSize(16, 16, N, M), as_gpu, bs_gpu, cs_gpu, M, K, N); };
        executor("matrix_multiplication_simple", "simple: ", func, benchmarkingIters, gflops);
        cs_gpu.readN(cs.data(), M * N);
    }
    int ret_code = check_correct(cs_cpu_reference, cs, N, M);
    if (ret_code) {
        return ret_code;
    }

    {
        auto func = [&](ocl::Kernel &k) { k.exec(gpu::WorkSize(16, 16, N, M), as_gpu, bs_gpu, cs_gpu, M, K, N); };
        executor("matrix_multiplication_tile", "local memory: ", func, benchmarkingIters, gflops);
        cs_gpu.readN(cs.data(), M * N);
    }
    ret_code = check_correct(cs_cpu_reference, cs, N, M);
    if (ret_code) {
        return ret_code;
    }

    {
        auto func = [&](ocl::Kernel &k) { k.exec(gpu::WorkSize(16, 4, N, M / 4), as_gpu, bs_gpu, cs_gpu, M, K, N); };
        executor("matrix_multiplication_more_thread_work", "more work per thread: ", func, benchmarkingIters, gflops);
        cs_gpu.readN(cs.data(), M * N);
    }
    ret_code = check_correct(cs_cpu_reference, cs, N, M);
    if (ret_code) {
        return ret_code;
    }

    return 0;
}
