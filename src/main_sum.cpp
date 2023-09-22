#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <numeric>

#include "cl/sum_cl.h"
#include "libgpu/work_size.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

struct Params {
    const std::string& device;
    std::vector<unsigned int>& as;
    unsigned int benchmarking_iters;
    unsigned int expect;
    unsigned int src_n;
    unsigned int res_n;
};

void sum(const gpu::WorkSize &work_size, const Params &params, const std::string& msg, const std::string &func) {
    ocl::Kernel adder(sum_kernel, sum_kernel_length, func);
    adder.compile();

    gpu::gpu_mem_32u src_gpu, res_gpu;
    src_gpu.resizeN(params.src_n);
    src_gpu.writeN(params.as.data(), params.src_n);

    res_gpu.resizeN(params.res_n);

    timer t;
    for (int iter = 0; iter < params.benchmarking_iters; ++iter) {
        std::vector<unsigned int> sum(params.res_n, 0);
        res_gpu.writeN(sum.data(), params.res_n);

        adder.exec(work_size, src_gpu, res_gpu, params.src_n);

        res_gpu.readN(sum.data(), params.res_n, 0);

        sum[0] = std::accumulate(sum.begin(), sum.end(), 0);

        EXPECT_THE_SAME(params.expect, sum[0], "the \"" + name + "\" method does not sum correctly!");
        t.nextLap();
    }
    std::cout << msg << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << msg << ": " << (params.src_n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    unsigned int benchmarkingIters = 10;
    unsigned int n = 100 * 1000 * 1000;

    unsigned int reference_sum = 0;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "summation on CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "summation on CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+ : sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "summation on CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "summation on CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        std::cout << std::endl;

        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        Params params{trimmed(device.name), as, benchmarkingIters, reference_sum, n, 1};

        sum(gpu::WorkSize(workGroupSize, global_work_size), params, "with global atomic add", "sum1");

        unsigned int temp = global_work_size / 128;
        sum(gpu::WorkSize(workGroupSize, temp), params, "with loop", "sum2");

        sum(gpu::WorkSize(workGroupSize, temp), params, "with loop and coalesced access", "sum3");

        sum(gpu::WorkSize(workGroupSize, global_work_size), params, "with local memory and global thread", "sum4");

        params.res_n = global_work_size / workGroupSize;
        sum(gpu::WorkSize(workGroupSize, global_work_size), params, "with tree", "sum5");
    }
}
