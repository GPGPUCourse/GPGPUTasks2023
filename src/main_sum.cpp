#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

int benchmarkingIters = 10;

void exec(const std::string &kernelName, const gpu::WorkSize &workingSize, const gpu::gpu_mem_32u &as_gpu,
          gpu::gpu_mem_32u &result, unsigned int n, unsigned int reference_sum) {
    ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernelName);
    kernel.compile();
    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        unsigned int sum = 0;
        result.writeN(&sum, 1);
        kernel.exec(workingSize, as_gpu, result, n);
        result.readN(&sum, 1);
        EXPECT_THE_SAME(sum, reference_sum, "GPU result should be consistent!");
        t.nextLap();
    }

    std::cout << "GPU " << kernelName << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU " << kernelName << ": " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

int main(int argc, char **argv) {
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
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
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
    }

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        unsigned int work_group_size = 128;
        unsigned int global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;

        gpu::gpu_mem_32u as_gpu, result;
        as_gpu.resizeN(n);
        result.resizeN(1);
        as_gpu.writeN(as.data(), n);

        exec("baseline", gpu::WorkSize(work_group_size, global_work_size), as_gpu, result, n, reference_sum);
        exec("cycle_3", gpu::WorkSize(work_group_size, (n + 2) / 3), as_gpu, result, n, reference_sum);
        exec("cycle_64", gpu::WorkSize(work_group_size, (n + 63) / 64), as_gpu, result, n, reference_sum);
        exec("cycle_coalesced_4", gpu::WorkSize(work_group_size, (n + 3) / 4), as_gpu, result, n, reference_sum);
        exec("cycle_coalesced_64", gpu::WorkSize(work_group_size, (n + 63) / 64), as_gpu, result, n, reference_sum);
        exec("local_mem", gpu::WorkSize(work_group_size, global_work_size), as_gpu, result, n, reference_sum);
        exec("tree", gpu::WorkSize(work_group_size, global_work_size), as_gpu, result, n, reference_sum);
    }
}
