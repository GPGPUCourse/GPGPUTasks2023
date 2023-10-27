#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
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

#define VALUES_PER_WORKITEM 32
#define WORKGROUP_SIZE 64

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

        unsigned int n = as.size();
        unsigned int work_group_size = 64;
        unsigned int global_work_size = (n + work_group_size - 1) / work_group_size * work_group_size;

        gpu::gpu_mem_32u as_gpu, sum_gpu;
        as_gpu.resizeN(n);
        sum_gpu.resizeN(1);
        as_gpu.writeN(as.data(), n);

        std::string defines =  "-DVALUES_PER_WORKITEM=" + std::to_string(VALUES_PER_WORKITEM) + " "
                + "-DWORKGROUP_SIZE=" + std::to_string(WORKGROUP_SIZE);

        std::cout << defines << std::endl;

        std::string kernel_name;
        {
            gpu::WorkSize workSize = gpu::WorkSize(work_group_size, global_work_size);
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_gpu_1", defines);
            bool printLog = true;
            kernel.compile(printLog);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                kernel.exec(workSize, as_gpu, sum_gpu, n);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU " << kernel_name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU " << kernel_name << ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        {
            gpu::WorkSize workSize = gpu::WorkSize(work_group_size,
            ((global_work_size + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM + work_group_size - 1) / work_group_size * work_group_size);
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_gpu_3", defines);
            bool printLog = false;
            kernel.compile(printLog);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                kernel.exec(workSize, as_gpu, sum_gpu, n);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU " << kernel_name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU " << kernel_name << ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        {
            gpu::WorkSize workSize = gpu::WorkSize(work_group_size,
            ((global_work_size + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM + work_group_size - 1) / work_group_size * work_group_size);
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_gpu_4", defines);
            bool printLog = false;
            kernel.compile(printLog);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                kernel.exec(workSize, as_gpu, sum_gpu, n);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU " << kernel_name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU " << kernel_name << ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        {
            gpu::WorkSize workSize = gpu::WorkSize(work_group_size, global_work_size);
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_gpu_5", defines);
            bool printLog = false;
            kernel.compile(printLog);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                kernel.exec(workSize, as_gpu, sum_gpu, n);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU " << kernel_name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU " << kernel_name << ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
        {
            gpu::WorkSize workSize = gpu::WorkSize(work_group_size, global_work_size);
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_gpu_6", defines);
            bool printLog = false;
            kernel.compile(printLog);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                unsigned int sum = 0;
                sum_gpu.writeN(&sum, 1);
                kernel.exec(workSize, as_gpu, sum_gpu, n);
                sum_gpu.readN(&sum, 1);
                EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
                t.nextLap();
            }
            std::cout << "GPU " << kernel_name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU " << kernel_name << ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
