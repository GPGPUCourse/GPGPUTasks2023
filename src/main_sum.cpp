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

void printStats(timer t, unsigned int n, std::string method) {
    std::cout << method << std::endl;
    std::cout << "  " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "  " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

void runKernel(std::string kernel_name,
               std::string method_name,
               unsigned int n,
               gpu::gpu_mem_32u* as_gpu,
               unsigned int real_sum,
               unsigned int work_group_size = 128,
               unsigned int global_work_size = 0,
               int benchmarkingIters = 10,
               unsigned int sum_buffer_size = 1) {
    ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);

    if (global_work_size == 0) {
        global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;
    }
    gpu::gpu_mem_32u sum_gpu;
    sum_gpu.resizeN(sum_buffer_size);
    std::vector<unsigned int> zeroes(sum_buffer_size, 0);
    unsigned int sum;
    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        sum_gpu.writeN(zeroes.data(), sum_buffer_size);
        kernel.exec(gpu::WorkSize(work_group_size, global_work_size),
                    *as_gpu, sum_gpu, n);
        sum_gpu.readN(&sum, 1);
        EXPECT_THE_SAME(real_sum, sum, method_name + " result should be consistent!");
        t.nextLap();
    }
    printStats(t, n, method_name);
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
        printStats(t, n, "CPU");
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
        printStats(t, n, "CPU OMP");
    }

    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);
        {
            runKernel("sum_baseline",
                      "GPU atomic add",
                      n,
                      &as_gpu,
                      reference_sum
            );
        }

        {
            unsigned int values_per_workitem = 8;
            unsigned int work_group_size = 128;
            unsigned int global_work_size = (n + values_per_workitem) / values_per_workitem;
            global_work_size = (global_work_size + work_group_size - 1) / work_group_size * work_group_size;
            runKernel("sum_looped",
                      "GPU loop",
                      n,
                      &as_gpu,
                      reference_sum,
                      work_group_size,
                      global_work_size
            );
            runKernel("sum_looped_coalesced",
                      "GPU coalesced loop",
                      n,
                      &as_gpu,
                      reference_sum,
                      work_group_size,
                      global_work_size
            );
        }

        {
            runKernel("sum_with_local",
                      "GPU local memory buffer",
                      n,
                      &as_gpu,
                      reference_sum,
                      128
            );
        }

        {
            runKernel("sum_tree",
                      "GPU tree",
                      n,
                      &as_gpu,
                      reference_sum
            );
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_tree2");
            unsigned int work_group_size = 128;
            gpu::gpu_mem_32u sum_gpu;
            gpu::gpu_mem_32u tmp_gpu;
            sum_gpu.resizeN(n);
            tmp_gpu.resizeN(n);
            std::vector<unsigned int> zeroes(n, 0);
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int n1 = n;
                unsigned int c = 0;
                while (n1 > 1) {
                    unsigned int global_work_size = (n1 + work_group_size - 1) / work_group_size * work_group_size;
                    if (c == 0) {
                        kernel.exec(gpu::WorkSize(work_group_size, global_work_size),
                                    as_gpu,
                                    tmp_gpu,
                                    n1);
                    }
                    else {
                        kernel.exec(gpu::WorkSize(work_group_size, global_work_size),
                                    tmp_gpu,
                                    sum_gpu,
                                    n1);
                        std::swap(tmp_gpu, sum_gpu);
                    }
                    n1 = global_work_size / work_group_size;
                    c++;
                    if (c == 5) { // chance to stop earlier
                        break;
                    }
                }
                if (c == 0) {
                    throw std::runtime_error("No actions made");
                }
                unsigned int sum = 0;
                std::vector<unsigned int> sumv(n1, 0);
                tmp_gpu.readN(sumv.data(), n1);
                for (unsigned int i = 0; i < n1; ++i) {
                    sum += sumv[i];
                }
                EXPECT_THE_SAME(reference_sum, sum, "GPU multiple trees result should be consistent!");
                t.nextLap();
            }
            printStats(t, n, "GPU multiple trees");
        }
    }
}
