#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

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

void sum(const gpu::WorkSize &work_size, const std::vector<unsigned int> as, const unsigned int src_n,
         const unsigned int res_n, const unsigned int benchmarkingIters, const unsigned int expect,
         const std::string &device, const std::string &name) {
    ocl::Kernel adder(sum_kernel, sum_kernel_length, name);
    adder.compile();

    gpu::gpu_mem_32u src_gpu, res_gpu;
    src_gpu.resizeN(src_n);
    src_gpu.writeN(as.data(), src_n);

    res_gpu.resizeN(res_n);

    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        std::vector<unsigned int> sum(res_n, 0);
        res_gpu.writeN(sum.data(), res_n);

        adder.exec(work_size, src_gpu, res_gpu, src_n);

        res_gpu.readN(sum.data(), 1, 0);
        EXPECT_THE_SAME(expect, sum[0], "the \"" + name + "\" method does not sum correctly!");
        t.nextLap();
    }
    std::cout << device << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << device << ": " << (src_n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    std::cout << std::endl;
}

int main(int argc, char **argv) {
    int benchmarkingIters = 10;
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
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
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
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        std::cout << std::endl;

        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        sum(gpu::WorkSize(workGroupSize, global_work_size), as, n, 1, benchmarkingIters, reference_sum,
            trimmed(device.name), "sum1");

        global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize / 64;
        sum(gpu::WorkSize(workGroupSize, global_work_size), as, n, 1, benchmarkingIters, reference_sum,
            trimmed(device.name), "sum2");

        sum(gpu::WorkSize(workGroupSize, global_work_size), as, n, 1, benchmarkingIters, reference_sum,
            trimmed(device.name), "sum3");
    }
}
