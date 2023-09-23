#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#define VALUES_PER_WORKITEM 32
#define WORKGROUP_SIZE 128

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
        // gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu, res_gpu;
        as_gpu.resizeN(n);
        res_gpu.resizeN(1);

        as_gpu.writeN(as.data(), n);

        unsigned int global_work_size = (n + 127) / 128 * 128;
        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_gpu_global_atomic");
            kernel.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                res_gpu.writeN(&sum, 1);
                kernel.exec(gpu::WorkSize(128, global_work_size), as_gpu, res_gpu, n);
                res_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU global_atomic: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU global_atomic: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_gpu_loop");
            kernel.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int res = 0;
                res_gpu.writeN(&res, 1);
                kernel.exec(gpu::WorkSize(128, global_work_size), as_gpu, res_gpu, n);
                res_gpu.readN(&res, 1);
                EXPECT_THE_SAME(reference_sum, res, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU loop: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU loop: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_gpu_loop_coalesced");
            kernel.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                res_gpu.writeN(&sum, 1);
                kernel.exec(gpu::WorkSize(128, global_work_size), as_gpu, res_gpu, n);
                res_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU loop_coalesced: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU loop_coalesced: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_gpu_local_mem");
            kernel.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                res_gpu.writeN(&sum, 1);
                kernel.exec(gpu::WorkSize(128, global_work_size), as_gpu, res_gpu, n);
                res_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU local_mem: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU local_mem: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_gpu_tree_atomic");
            kernel.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int res = 0;
                res_gpu.writeN(&res, 1);
                kernel.exec(gpu::WorkSize(128, global_work_size), as_gpu, res_gpu, n);
                res_gpu.readN(&res, 1);
                EXPECT_THE_SAME(reference_sum, res, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU tree_atomic: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU tree_atomic: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
