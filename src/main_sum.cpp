#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <numeric>

#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000 + 35;
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
            EXPECT_THE_SAME(reference_sum, sum, "CPU: result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OMP: result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

    auto benchmark_sum_gpu = [&](
                                     const std::string& kernel_name,
                                     const std::string& method_name,
                                     const unsigned int& less_global_work_size,
                                     unsigned int number_of_sums_joined_in_cpu,
                                     const bool& compileLog)
    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);

        ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);
        kernel.compile(compileLog);

        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        global_work_size = (global_work_size + less_global_work_size - 1) / less_global_work_size;
        global_work_size = (global_work_size + workGroupSize - 1) / workGroupSize * workGroupSize;

        if (number_of_sums_joined_in_cpu == -1)
            number_of_sums_joined_in_cpu = global_work_size / workGroupSize;

        gpu::gpu_mem_32u sum_gpu;
        sum_gpu.resizeN(number_of_sums_joined_in_cpu);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            std::vector<unsigned int> sums(number_of_sums_joined_in_cpu, 0);
            sum_gpu.writeN(sums.data(), number_of_sums_joined_in_cpu);

            kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        as_gpu, n, sum_gpu);

            sum_gpu.readN(sums.data(), number_of_sums_joined_in_cpu);
            sum = std::accumulate(sums.begin(), sums.end(), 0u);

            EXPECT_THE_SAME(reference_sum, sum, "GPU: result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU " << method_name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU " << method_name << ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    };

    benchmark_sum_gpu("sum1", "atomic_add", 1, 1, false);
    benchmark_sum_gpu("sum2", "ranges", 128, 1, false);
    benchmark_sum_gpu("sum3", "coalesced_groups", 128, 1, false);
    benchmark_sum_gpu("sum4", "local_memory", 1, 1, false);
    benchmark_sum_gpu("sum5", "tree_local_memory", 1, 1, false);

    benchmark_sum_gpu("sum6", "tree_local_global_memory", 1, -1, false);
    benchmark_sum_gpu("sum3_global_mem", "coalesced_groups_global_memory", 1, 128, false);
}
