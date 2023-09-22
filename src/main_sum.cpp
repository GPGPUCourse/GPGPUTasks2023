#include "cl/sum_cl.h"
#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void run_kernel_bench(std::string name, gpu::gpu_mem_32u &gpu_result,
                      gpu::gpu_mem_32u &data, gpu::WorkSize size,
                      unsigned int reference_sum, int benchmarkingIters, int n) {
    ocl::Kernel kernel(sum_kernel, sum_kernel_length, name);
    kernel.compile();

    timer t;
    for (int iter = 0; iter < benchmarkingIters; iter++) {
        unsigned int res = 0;
        gpu_result.writeN(&res, 1);
        kernel.exec(size, gpu_result, data, n);
        gpu_result.readN(&res, 1);
        EXPECT_THE_SAME(reference_sum, res, "sum on GPU is wrong");
        t.nextLap();
    }
    std::cout << name << " " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << name << " " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
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
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(as.size());
        as_gpu.writeN(as.data(), as.size());

        gpu::gpu_mem_32u gpu_result;
        gpu_result.resizeN(1);

        {
            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            run_kernel_bench("only_atomic", gpu_result, as_gpu, gpu::WorkSize(workGroupSize, global_work_size),
                             reference_sum, benchmarkingIters, n);
        }
        {
            unsigned int workGroupSize = 128;
            unsigned int global_work_size = ((as.size() + 128 - 1) / 128 + workGroupSize - 1) / workGroupSize * workGroupSize;
            run_kernel_bench("cycle", gpu_result, as_gpu, gpu::WorkSize(workGroupSize, global_work_size),
                             reference_sum, benchmarkingIters, n);
        }
        {
            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            run_kernel_bench("coalesced", gpu_result, as_gpu, gpu::WorkSize(workGroupSize, global_work_size),
                             reference_sum, benchmarkingIters, n);
        }
        {
            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            run_kernel_bench("mem_local", gpu_result, as_gpu, gpu::WorkSize(workGroupSize, global_work_size),
                             reference_sum, benchmarkingIters, n);
        }
        {
            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            run_kernel_bench("tree", gpu_result, as_gpu, gpu::WorkSize(workGroupSize, global_work_size),
                             reference_sum, benchmarkingIters, n);
        }
    }
}
