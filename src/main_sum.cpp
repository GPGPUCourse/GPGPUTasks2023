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
         std::vector<std::string> kernel_names =
         {
             "sum_atomic",
             "sum_atomic_cycle",
             "sum_atomic_cycle_coalesced",
             "sum_local",
             "sum_tree_atomic",
             //"sum_tree_array"  <-- здесь в конце нужен массив, потому вместе со всеми не получится
         };
         gpu::gpu_mem_32u as_gpu;
         as_gpu.resizeN(n);
         as_gpu.writeN(as.data(), n);
         gpu::gpu_mem_32u result;
         result.resizeN(1);

         for (const std::string& kernel_name : kernel_names)
         {
             ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernel_name);
             bool printLog = false;
             kernel.compile(printLog);

             timer t;
             for (int iter = 0; iter < benchmarkingIters; iter++)
             {
                 unsigned int sum = 0;
                 result.writeN(&sum, 1);
                 kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, n), as_gpu, result, n);
                 result.readN(&sum, 1);
                 EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent: " + kernel_name);
                 t.nextLap();
             }
             std::cout << "GPU " << kernel_name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
             std::cout << "GPU " << kernel_name << ": " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
         }

         ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_tree_array");
         bool printLog = false;
         kernel.compile(printLog);

         gpu::gpu_mem_32u result_arr;
         size_t n_groups = gpu::divup(n, WORKGROUP_SIZE);
         std::vector<unsigned int> vec(n_groups, 0);
         result_arr.resizeN(n_groups);

         timer t;
         for (int iter = 0; iter < benchmarkingIters; iter++)
         {
             t.stop();
             unsigned int sum = 0;
             vec.assign(n_groups, 0);
             result_arr.writeN(vec.data(), n_groups);

             t.start();
             kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, n), as_gpu, result_arr, n);
             t.stop();

             result_arr.readN(vec.data(), n_groups);
             sum = std::accumulate(vec.begin(), vec.end(), 0U);
             EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
             t.nextLap();
         }
         std::cout << "GPU sum_tree_array: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
         std::cout << "GPU sum_tree_array: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    return 0;
}
