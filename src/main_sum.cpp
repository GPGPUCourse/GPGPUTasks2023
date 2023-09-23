#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>


#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
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

    // {
    //     timer t;
    //     for (int iter = 0; iter < benchmarkingIters; ++iter) {
    //         unsigned int sum = 0;
    //         for (int i = 0; i < n; ++i) {
    //             sum += as[i];
    //         }
    //         EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
    //         t.nextLap();
    //     }
    //     std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    //     std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    // }

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
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_global_atomic");

            gpu::gpu_mem_32u sum_gpu;
            gpu::gpu_mem_32u as_gpu;
            sum_gpu.resizeN(1);
            as_gpu.resizeN(n);
            as_gpu.writeN(as.data(), n);
            kernel.compile();

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = n;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), sum_gpu, as_gpu, n);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU global atomic add:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU global atomic add:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s"
                      << std::endl;
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_for");

            gpu::gpu_mem_32u sum_gpu;
            gpu::gpu_mem_32u as_gpu;
            sum_gpu.resizeN(1);
            as_gpu.resizeN(n);
            as_gpu.writeN(as.data(), n);
            kernel.compile();


            unsigned int workGroupSize = 128;
            unsigned int values_per_workitem = 64;
            unsigned int global_work_size = (n + values_per_workitem) / values_per_workitem;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), sum_gpu, as_gpu, n);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU for loop:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU for loop:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_for_coalesed");

            gpu::gpu_mem_32u sum_gpu;
            gpu::gpu_mem_32u as_gpu;
            sum_gpu.resizeN(1);
            as_gpu.resizeN(n);
            as_gpu.writeN(as.data(), n);
            kernel.compile();


            unsigned int workGroupSize = 128;
            unsigned int values_per_workitem = 64;
            unsigned int global_work_size = (n + values_per_workitem) / values_per_workitem;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), sum_gpu, as_gpu, n);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU for loop coalesed:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU for loop coalesed:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s"
                      << std::endl;
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_local_memory");

            gpu::gpu_mem_32u sum_gpu;
            gpu::gpu_mem_32u as_gpu;
            sum_gpu.resizeN(1);
            as_gpu.resizeN(n);
            as_gpu.writeN(as.data(), n);
            kernel.compile();

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize) / workGroupSize * workGroupSize;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), sum_gpu, as_gpu, n);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU local memory:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU local memory:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_tree");

            gpu::gpu_mem_32u sum_gpu;
            gpu::gpu_mem_32u as_gpu;
            sum_gpu.resizeN(1);
            as_gpu.resizeN(n);
            as_gpu.writeN(as.data(), n);
            kernel.compile();

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize) / workGroupSize * workGroupSize;
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), sum_gpu, as_gpu, n);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU tree:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU tree:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
