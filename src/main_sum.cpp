#include "cl/sum_cl.h"
#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#define WORK_GROUP_SIZE 128
#define VALUES_PER_WORKITEM 128

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
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
        std::cout << "[CPU]: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "[CPU]: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
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
        std::cout << "[CPU OMP]: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "[CPU OMP]: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    // Choose device.
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    // Create context.
    gpu::Context ctx;
    ctx.init(device.device_id_opencl);
    ctx.activate();

    // Create global buffer for array.
    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(as.size());
    as_gpu.writeN(as.data(), as.size());

    gpu::gpu_mem_32u res_gpu;
    res_gpu.resizeN(1);

    auto gpu_bench = [&](const std::string &kernel_name, gpu::WorkSize sz) {
        // Create and compile kernel.
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);
        kernel.compile(false);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int result = 0;
            // Set result to 0.
            res_gpu.writeN(&result, 1);
            kernel.exec(sz, as_gpu, res_gpu, n);
            res_gpu.readN(&result, 1);
            EXPECT_THE_SAME(reference_sum, result, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "[" << kernel_name << "]: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "[" << kernel_name << "]: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    };

    {
        unsigned int workGroupSize = WORK_GROUP_SIZE;
        unsigned int globalWorkSize = (as.size() + workGroupSize - 1) / workGroupSize * workGroupSize;
        gpu_bench("baseline", gpu::WorkSize(workGroupSize, globalWorkSize));
    }

    {
        unsigned int workGroupSize = WORK_GROUP_SIZE;
        unsigned int needToCover = (as.size() + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM;
        unsigned int globalWorkSize = (needToCover + workGroupSize - 1) / workGroupSize * workGroupSize;
        gpu_bench("cycle", gpu::WorkSize(workGroupSize, globalWorkSize));
    }

    {
        unsigned int workGroupSize = WORK_GROUP_SIZE;
        unsigned int needToCover = (as.size() + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM;
        unsigned int globalWorkSize = (needToCover + workGroupSize - 1) / workGroupSize * workGroupSize;
        gpu_bench("cycle_coalesced", gpu::WorkSize(workGroupSize, globalWorkSize));
    }

    // Necessarily use WORK_GROUP_SIZE below.
    {
        unsigned int workGroupSize = WORK_GROUP_SIZE;
        unsigned int globalWorkSize = (as.size() + workGroupSize - 1) / workGroupSize * workGroupSize;
        gpu_bench("local_mem", gpu::WorkSize(workGroupSize, globalWorkSize));
    }

    {
        unsigned int workGroupSize = WORK_GROUP_SIZE;
        unsigned int globalWorkSize = (as.size() + workGroupSize - 1) / workGroupSize * workGroupSize;
        gpu_bench("tree", gpu::WorkSize(workGroupSize, globalWorkSize));
    }
}
