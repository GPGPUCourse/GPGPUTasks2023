#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

#define VALUES_PER_WORK_ITEM 32

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

        gpu::gpu_mem_32u as_buffer;
        gpu::gpu_mem_32u sum_buffer;

        as_buffer.resizeN(n);
        sum_buffer.resizeN(1);

        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        unsigned int n_work_groups = global_work_size / workGroupSize;

        as_buffer.writeN(as.data(), n);
        const unsigned int init = 0;

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "globalAtomSum");
            kernel.compile();

            timer t;
            unsigned int sum = 0;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum_buffer.writeN(&init, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                            as_buffer, n, sum_buffer);
                t.nextLap();
            }
            sum_buffer.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            std::cout << "GPU (atomic sum): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (atomic sum): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "loopSum");
            kernel.compile();

            unsigned int currentWorkGroupSize = 128;
            unsigned int current_global_work_size = (((n + VALUES_PER_WORK_ITEM - 1) / VALUES_PER_WORK_ITEM) + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            unsigned int sum = 0;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum_buffer.writeN(&init, 1);
                kernel.exec(gpu::WorkSize(currentWorkGroupSize, current_global_work_size),
                            as_buffer, n, sum_buffer);
                t.nextLap();
            }
            sum_buffer.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            std::cout << "GPU (loop sum): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (loop sum): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "loopCoalescedSum");
            kernel.compile();


            unsigned int currentWorkGroupSize = 128;
            unsigned int current_global_work_size = (((n + VALUES_PER_WORK_ITEM - 1) / VALUES_PER_WORK_ITEM) + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            unsigned int sum = 0;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum_buffer.writeN(&init, 1);
                kernel.exec(gpu::WorkSize(currentWorkGroupSize, current_global_work_size),
                            as_buffer, n, sum_buffer);
                t.nextLap();
            }
            sum_buffer.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            std::cout << "GPU (loop coalesced sum): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (loop coalesced sum): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sumWithLocalMemes");
            kernel.compile();

            timer t;
            unsigned int sum = 0;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum_buffer.writeN(&init, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                            as_buffer, n, sum_buffer);
                t.nextLap();
            }
            sum_buffer.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            std::cout << "GPU (Sum With Local Memes): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (Sum With Local Memes): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "treeSum");
            kernel.compile();

            timer t;
            unsigned int sum = 0;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                sum_buffer.writeN(&init, 1);
                kernel.exec(gpu::WorkSize(workGroupSize, global_work_size),
                            as_buffer, n, sum_buffer);
                t.nextLap();
            }
            sum_buffer.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            std::cout << "GPU (tree sum): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (tree sum): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
