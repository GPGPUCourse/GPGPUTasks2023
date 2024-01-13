#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"
#include <numeric>

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

#define WORKGROUP_SIZE 64
    {
        // TODO: implement on OpenCL
         gpu::Device device = gpu::chooseGPUDevice(argc, argv);
         gpu::Context context;
         context.init(device.device_id_opencl);
         context.activate();

         struct KernelConfig
         {
             std::string name;
             size_t n_groups;
         };

         std::vector<KernelConfig> kernel_configs =
         {
             { "sum_atomic", n },
             { "sum_atomic_cycle", gpu::divup(n, 64) },
             { "sum_atomic_cycle_coalesced", gpu::divup(n, 64) },
             { "sum_local", n },
             { "sum_tree_atomic", n },
             //"sum_tree_array"  <-- здесь в конце нужен массив, потому вместе со всеми не получится
         };
         gpu::gpu_mem_32u as_gpu;
         as_gpu.resizeN(n);
         as_gpu.writeN(as.data(), n);
         gpu::gpu_mem_32u result;
         result.resizeN(1);

         for (const auto& kernel_conf : kernel_configs)
         {
             const std::string& kernel_name = kernel_conf.name;
             const size_t n_groups = kernel_conf.n_groups;
             ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);
             bool printLog = false;
             kernel.compile(printLog);

             timer t;
             for (int iter = 0; iter < benchmarkingIters; iter++)
             {
                 unsigned int sum = 0;
                 result.writeN(&sum, 1);
                 kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, n_groups), as_gpu, result, n);
                 result.readN(&sum, 1);
                 EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent: " + kernel_name);
                 t.nextLap();
             }
             std::cout << "GPU " << kernel_name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
             std::cout << "GPU " << kernel_name << ": " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
         }

#define THRESHOLD 100000
         ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_tree_array");
         bool printLog = false;
         kernel.compile(printLog);

         gpu::gpu_mem_32u result_arr1;
         gpu::gpu_mem_32u result_arr2;
         std::vector<unsigned int> vec(THRESHOLD, 0);
         result_arr1.resizeN(n);
         result_arr2.resizeN(n);

         timer t;
         for (int iter = 0; iter < benchmarkingIters; iter++)
         {
             t.stop();
             unsigned int sum = 0;
             result_arr1.writeN(as.data(), n);
             vec.assign(THRESHOLD, 0);
             unsigned int curr_n = n;

             t.start();
             for (;curr_n > THRESHOLD; curr_n = gpu::divup(curr_n, WORKGROUP_SIZE))
             {
                 kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, curr_n), result_arr1, result_arr2, curr_n);
                 result_arr1.swap(result_arr2);
             }
             result_arr1.readN(vec.data(), curr_n);
             sum = std::accumulate(vec.begin(), vec.end(), 0U);
             t.stop();

             EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
             t.nextLap();
         }
         std::cout << "GPU sum_tree_array: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
         std::cout << "GPU sum_tree_array: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    return 0;
}
