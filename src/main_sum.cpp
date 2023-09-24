#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

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
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu, result_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);
        result_gpu.resizeN(n);
        std::vector<unsigned int> zero(n);

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_atomic");
            kernel.compile();

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                result_gpu.writeN(zero.data(), 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                    as_gpu, result_gpu, n);
                
                unsigned int sum = 0;
                result_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU (atomic operations) result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU (atomic operations): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (atomic operations): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_atomic_loop");
            kernel.compile();

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                result_gpu.writeN(zero.data(), 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                    as_gpu, result_gpu, n);
                unsigned int sum = 0;
                result_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU (atomic loop) result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU (atomic loop): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (atomic loop): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_coalesced_loop");
            kernel.compile();

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                result_gpu.writeN(zero.data(), 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                    as_gpu, result_gpu, n);
                unsigned int sum = 0;
                result_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU (coalesced loop) result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU (coalesced loop): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (coalesced loop): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_local_memory");
            kernel.compile();

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                result_gpu.writeN(zero.data(), 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                    as_gpu, result_gpu, n);
                unsigned int sum = 0;
                result_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU (local memory) result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU (local memory): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (local memory): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_tree");
            kernel.compile();

            unsigned int workGroupSize = 128;

            gpu::gpu_mem_32u tree_input_gpu;
            tree_input_gpu.resizeN(n);
            tree_input_gpu.writeN(as.data(), n);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                for (int lvl_size = n; lvl_size > 1; lvl_size = (lvl_size + workGroupSize - 1) / workGroupSize) {
                    result_gpu.writeN(zero.data(), lvl_size);
                    unsigned int global_work_size = (lvl_size + workGroupSize - 1) / workGroupSize * workGroupSize;
                    kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                        tree_input_gpu, result_gpu, n);
                    result_gpu.copyToN(tree_input_gpu, lvl_size);
                }
                unsigned int sum = 0;
                result_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU (tree) result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU (tree): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (tree): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
