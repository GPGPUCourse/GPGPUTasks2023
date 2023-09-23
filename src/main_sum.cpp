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

#define DIV_ROUND_UP(a, b) (((a) + (b) - 1) / (b))

// keep synced with same variable in sum.cl
#define VALUES_PER_WORKITEM 64
#define WORKGROUP_SIZE 64

// should executed inside context
void run_gpu_kernel(std::vector<unsigned int> data, unsigned int expected, const char* kernelName, size_t benchmarkingIters,
                    unsigned int workGroupSize, unsigned int globalWorkSize)
{
    timer t;

    const unsigned int n = data.size();

    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u result_vram;
    as_gpu.resizeN(n);
    result_vram.resizeN(1);

    ocl::Kernel kernel(sum_kernel, sum_kernel_length, kernelName);
    kernel.compile();


    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        as_gpu.writeN(data.data(), n);
        unsigned int result = 0;
        result_vram.writeN(&result, 1);

        kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), as_gpu, result_vram, n);
            
        result_vram.readN(&result, 1);
        EXPECT_THE_SAME(expected, result, "GPU result should be consistent!");

        t.nextLap();
    }

    std::cout << "GPU(" << kernelName << "): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU(" << kernelName << "): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

int main(int argc, char **argv)
{
    int benchmarkingIters = 10;

    unsigned int referenceSum = 0;
    unsigned int n = 100*1000*1000;
    /* unsigned int n = 1*1000*1000; */
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        referenceSum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(referenceSum, sum, "CPU result should be consistent!");
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
            EXPECT_THE_SAME(referenceSum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        const unsigned int workGroupWork = WORKGROUP_SIZE * VALUES_PER_WORKITEM;

        run_gpu_kernel(as, referenceSum, "sum_baseline", benchmarkingIters,
                WORKGROUP_SIZE, DIV_ROUND_UP(n, WORKGROUP_SIZE) * WORKGROUP_SIZE);

        run_gpu_kernel(as, referenceSum, "sum_loop", benchmarkingIters,
                WORKGROUP_SIZE, DIV_ROUND_UP(n, workGroupWork) * WORKGROUP_SIZE);

        run_gpu_kernel(as, referenceSum, "sum_loop_coalesced", benchmarkingIters,
                WORKGROUP_SIZE, DIV_ROUND_UP(n, workGroupWork) * WORKGROUP_SIZE);

        run_gpu_kernel(as, referenceSum, "sum_local_mem", benchmarkingIters,
                WORKGROUP_SIZE, DIV_ROUND_UP(n, WORKGROUP_SIZE) * WORKGROUP_SIZE);

        // sum_tree special call
        {
            timer t;

            gpu::gpu_mem_32u as_gpu;
            gpu::gpu_mem_32u bs_gpu;
            as_gpu.resizeN(n);
            bs_gpu.resizeN(n);

            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_tree");
            kernel.compile();

            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                bs_gpu.writeN(as.data(), n);

                for (unsigned int alive = n; alive > 1; alive = DIV_ROUND_UP(alive, WORKGROUP_SIZE)) {
                    kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, DIV_ROUND_UP(alive, WORKGROUP_SIZE) * WORKGROUP_SIZE), as_gpu, bs_gpu, alive);
                    std::swap(as_gpu, bs_gpu);
                }
                    
                unsigned int result;
                as_gpu.readN(&result, 1);
                EXPECT_THE_SAME(referenceSum, result, "GPU result should be consistent!");

                t.nextLap();
            }

            std::cout << "GPU(sum_tree): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU(sum_tree): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
    return 0;
}
