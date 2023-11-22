#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <string>

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
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

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
        std::cout << "CPU:     \t\t" << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     \t\t" << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
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
        std::cout << "CPU OMP: \t\t" << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: \t\t" << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        uint sum_gpu = 0;
        const uint wgSize = 128;
        std::vector<unsigned int> zeros(1, 0);

        ocl::Kernel baseline(sum_kernel, sum_kernel_length, "sum_atomic");
        ocl::Kernel loop(sum_kernel, sum_kernel_length, "sum_loop"); // 51648.4
        ocl::Kernel loop_coalesced(sum_kernel, sum_kernel_length, "sum_loop_coalesced");
        ocl::Kernel local(sum_kernel, sum_kernel_length, "sum_local");
        ocl::Kernel tree(sum_kernel, sum_kernel_length, "sum_tree");

        ocl::Kernel kernels[5] = {baseline, loop, loop_coalesced, local, tree};
        uint totalWorks[] = {n, n /4, n / 4, n, n};
        std::string kernelNames[] = {" (baseline)\t\t", " (loop)\t\t", " (loop_coal)\t\t", " (local_mem)\t\t", " (tree)\t\t"};
        

        gpu::gpu_mem_32u as_gpu, result;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);
        result.resizeN(1);

        for (int kernelIdx = 0; kernelIdx < 5; ++kernelIdx) {
            timer t;
            for (int i = 0; i < benchmarkingIters; ++i) {
                result.writeN(zeros.data(), 1);
                kernels[kernelIdx].exec(gpu::WorkSize(wgSize, totalWorks[kernelIdx]), as_gpu, result, n);

                result.readN(&sum_gpu, 1);
                EXPECT_THE_SAME(reference_sum, sum_gpu, "CPU & GPU results must be the same!");
                t.nextLap();
            }

            std::cout << "GPU" << kernelNames[kernelIdx] << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU" << kernelNames[kernelIdx] << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
