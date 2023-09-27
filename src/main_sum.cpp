#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include "libgpu/context.h"
#include "cl/sum_cl.h"
#include <libgpu/shared_device_buffer.h>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


template <typename KERNEL_NAME_TYPE, typename KERNEL_LENGTH_TYPE, typename ARRAY_TYPE>
void runSumKernel(KERNEL_NAME_TYPE kernelName, KERNEL_LENGTH_TYPE kernelLength, const std::string& kernel_func_name,  const ARRAY_TYPE& arr, unsigned int n, int work_size, int global_work_size, int benchmarkingIters = 20) {
    ocl::Kernel sumKernel(kernelName, kernelLength , kernel_func_name);
    sumKernel.compile();
    gpu::gpu_mem_32u res;
    res.resizeN(1);

    timer t;
    unsigned int sum;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        sum = 0;
        res.writeN(&sum, 1);
        sumKernel.exec(gpu::WorkSize(work_size, global_work_size), arr, res, n);
        t.nextLap();
    }
    res.readN(&sum, 1);
    std::cout << "KERNEL NAME: " << kernel_func_name << std::endl;
    std::cout << "SUM GPU:     " << sum << std::endl;
    std::cout << "GPU:         " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU:         " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

int main(int argc, char **argv)
{
    int benchmarkingIters = 20;

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
        unsigned int sum;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "SUM:     " << sum << std::endl;
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        unsigned int sum;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "SUM OMP:     " << sum << std::endl;
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u arrBuff;
        arrBuff.resizeN(n);
        arrBuff.writeN(as.data(), n);

        auto get_global_size = [](unsigned int n, int workSize) {
            return (n + workSize - 1) / workSize * workSize;
        };
        int workSize = 256, globalWorkSize = get_global_size(n, workSize);
        runSumKernel(sum_kernel, sum_kernel_length, "sum_gpu_1", arrBuff, n, workSize, globalWorkSize);
        workSize = 256;
        globalWorkSize = (n + workSize - 1) / workSize;
        runSumKernel(sum_kernel, sum_kernel_length, "sum_gpu_2", arrBuff, n, workSize, globalWorkSize);
        runSumKernel(sum_kernel, sum_kernel_length, "sum_gpu_3", arrBuff, n, workSize, globalWorkSize);
        workSize = 128;
        globalWorkSize = get_global_size(n, workSize);
        runSumKernel(sum_kernel, sum_kernel_length, "sum_gpu_4", arrBuff, n, workSize, globalWorkSize);
        workSize = 64;
        globalWorkSize = get_global_size(n, workSize);
        runSumKernel(sum_kernel, sum_kernel_length, "sum_gpu_5", arrBuff, n, workSize, globalWorkSize);
    }
}
