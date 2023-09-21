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

const unsigned int zero = 0;

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
        gpu::gpu_mem_32u as_gpu, bs_gpu, res_gpu;

        unsigned int global_work_size = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;

        as_gpu.resizeN(n);
        bs_gpu.resizeN(n);
        res_gpu.resizeN(1);
        as_gpu.writeN(as.data(), n);

        {
            ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum_gpu_global_atomic");
            sum.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int res = 0;
                res_gpu.writeN(&zero, 1);
                sum.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size),
                        as_gpu, res_gpu, n);
                res_gpu.readN(&res, 1);
                EXPECT_THE_SAME(reference_sum, res, "GPU(sum_global_atomic) result should be constistent!");
                t.nextLap();
            }
            std::cout << "GPU global atomic: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU global atomic: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;

        }

        {
            ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum_gpu_cycle");
            sum.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int res = 0;
                res_gpu.writeN(&zero, 1);
                sum.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size),
                        as_gpu, res_gpu, n);
                res_gpu.readN(&res, 1);
                EXPECT_THE_SAME(reference_sum, res, "GPU(sum_cycle) result should be constistent!");
                t.nextLap();
            }
            std::cout << "GPU cycle: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU cycle: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum_gpu_cycle_coalesced");
            sum.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int res = 0;
                res_gpu.writeN(&zero, 1);
                sum.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size),
                        as_gpu, res_gpu, n);
                res_gpu.readN(&res, 1);
                EXPECT_THE_SAME(reference_sum, res, "GPU(sum_cycle_coalesced) result should be constistent!");
                t.nextLap();
            }
            std::cout << "GPU cycle coalesced: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU cycle coalesced: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum_gpu_local_memory");
            sum.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int res = 0;
                res_gpu.writeN(&zero, 1);
                sum.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size),
                        as_gpu, res_gpu, n);
                res_gpu.readN(&res, 1);
                EXPECT_THE_SAME(reference_sum, res, "GPU(sum_local_memory) result should be constistent!");
                t.nextLap();
            }
            std::cout << "GPU local memory: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU local memory: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

       {
            ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum_gpu_atomic_tree");
            sum.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int res = 0;
                res_gpu.writeN(&zero, 1);
                sum.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size),
                        as_gpu, res_gpu, n);
                res_gpu.readN(&res, 1);
                EXPECT_THE_SAME(reference_sum, res, "GPU(sum_atomic_tree) result should be constistent!");
                t.nextLap();
            }
            std::cout << "GPU atomic tree: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU atomic tree: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }


       {
            ocl::Kernel sum(sum_kernel, sum_kernel_length, "sum_gpu_tree");
            sum.compile();

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int res = 0;
                for (int current_n = n; current_n > 1; current_n = (current_n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) {
                    sum.exec(gpu::WorkSize(WORKGROUP_SIZE, global_work_size),
                            as_gpu, bs_gpu, current_n);
                    bs_gpu.copyToN(as_gpu, current_n);
                }
                as_gpu.readN(&res, 1);
                EXPECT_THE_SAME(reference_sum, res, "GPU(sum_tree) result should be constistent!");
                t.nextLap();
            }
            std::cout << "GPU tree: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU tree: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

    }
}
