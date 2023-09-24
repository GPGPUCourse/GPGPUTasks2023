#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "cl/sum_cl.h"
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#define VALUES_PER_WORKITEM 64
#define WORKGROUP_SIZE 64

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

unsigned int division(unsigned int a, unsigned int b) {
    return (a + b - 1) / b;
}

void run_tree(unsigned int n, int benchmarkingIters, const std::vector<unsigned int> &as, unsigned int reference_sum) {
    timer t;

    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u bs_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);

    ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_tree");
    kernel.compile();

    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        as_gpu.writeN(as.data(), n);
        bs_gpu.writeN(as.data(), n);

        unsigned int offset = n;
        while (offset > 1) {
            kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, division(offset, WORKGROUP_SIZE) * WORKGROUP_SIZE), as_gpu,
                        bs_gpu, offset);
            std::swap(as_gpu, bs_gpu);
            offset = division(offset, WORKGROUP_SIZE);
        }

        unsigned int result;
        as_gpu.readN(&result, 1);
        EXPECT_THE_SAME(reference_sum, result, "GPU result should be consistent!");

        t.nextLap();
    }

    std::cout << "GPU(sum_tree): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU(sum_tree): " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

void run_kernel(std::string kernel_name, const std::vector<unsigned int> &data, unsigned int reference_sum,
                size_t benchmarkingIters, unsigned int workGroupSize, unsigned int taskSize) {
    timer t;

    const unsigned int n = data.size();

    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u vram;
    as_gpu.resizeN(n);
    vram.resizeN(1);

    ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name.data());
    kernel.compile();

    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        as_gpu.writeN(data.data(), n);
        unsigned int result = 0;
        vram.writeN(&result, 1);

        kernel.exec(gpu::WorkSize(workGroupSize, taskSize), as_gpu, vram, n);

        vram.readN(&result, 1);
        EXPECT_THE_SAME(reference_sum, result, "GPU result should be consistent!");

        t.nextLap();
    }

    std::cout << "GPU(" << kernel_name << "): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU(" << kernel_name << "): " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

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

        const unsigned int taskSize = WORKGROUP_SIZE * VALUES_PER_WORKITEM;

        run_kernel("sum_baseline", as, reference_sum, benchmarkingIters, WORKGROUP_SIZE,
                   division(n, WORKGROUP_SIZE) * WORKGROUP_SIZE);

        run_kernel("sum_looped", as, reference_sum, benchmarkingIters, WORKGROUP_SIZE,
                   division(n, taskSize) * WORKGROUP_SIZE);

        run_kernel("sum_looped_coalesced", as, reference_sum, benchmarkingIters, WORKGROUP_SIZE,
                   division(n, taskSize) * WORKGROUP_SIZE);

        run_kernel("sum_local", as, reference_sum, benchmarkingIters, WORKGROUP_SIZE,
                   division(n, WORKGROUP_SIZE) * WORKGROUP_SIZE);
        run_tree(n, benchmarkingIters, as, reference_sum);
    }
}
