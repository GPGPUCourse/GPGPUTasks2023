#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

#define VALUES_PER_ITEM 32

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
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu;
        gpu::gpu_mem_32u res_gpu;

        as_gpu.resizeN(n);
        res_gpu.resizeN(1);
        
        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        unsigned int n_work_groups = global_work_size / workGroupSize;

        as_gpu.writeN(as.data(), n);

        {
            ocl::Kernel sum_base(sum_kernel, sum_kernel_length, "sumBase");
            sum_base.compile();

            timer t;
            unsigned int initialValue = 0, sum = 0;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                res_gpu.writeN(&initialValue, 1);
                sum_base.exec(gpu::WorkSize(workGroupSize, global_work_size),
                            as_gpu, res_gpu, n);
                t.nextLap();
            }
            res_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            std::cout << "GPU (base): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (base): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel sum_loop(sum_kernel, sum_kernel_length, "sumLoop");
            sum_loop.compile();

            timer t;
            unsigned int initialValue = 0, sum = 0;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                res_gpu.writeN(&initialValue, 1);
                sum_loop.exec(gpu::WorkSize(workGroupSize, global_work_size / VALUES_PER_ITEM),
                            as_gpu, res_gpu, n);
                t.nextLap();
            }
            res_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            std::cout << "GPU (loop): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (loop): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel sum_loop_coalesced(sum_kernel, sum_kernel_length, "sumLoopCoalesced");
            sum_loop_coalesced.compile();

            timer t;
            unsigned int initialValue = 0, sum = 0;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                res_gpu.writeN(&initialValue, 1);
                sum_loop_coalesced.exec(gpu::WorkSize(workGroupSize, global_work_size / VALUES_PER_ITEM),
                            as_gpu, res_gpu, n);
                t.nextLap();
            }
            res_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            std::cout << "GPU (coalesced loop): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (coalesced loop): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel sum_main_thread(sum_kernel, sum_kernel_length, "sumWithMainThread");
            sum_main_thread.compile();

            timer t;
            unsigned int initialValue = 0, sum = 0;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                res_gpu.writeN(&initialValue, 1);
                sum_main_thread.exec(gpu::WorkSize(workGroupSize, global_work_size),
                            as_gpu, res_gpu, n);
                t.nextLap();
            }
            res_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            std::cout << "GPU (With main thread): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (With main thread): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            ocl::Kernel sum_tree(sum_kernel, sum_kernel_length, "sumTree");
            sum_tree.compile();

            timer t;
            unsigned int initialValue = 0, sum = 0;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                res_gpu.writeN(&initialValue, 1);
                sum_tree.exec(gpu::WorkSize(workGroupSize, global_work_size),
                            as_gpu, res_gpu, n);
                t.nextLap();
            }
            res_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            std::cout << "GPU (Tree): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (Tree): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            gpu::gpu_mem_32u bs_gpu;
            bs_gpu.resizeN(n);
            res_gpu.resizeN(n);

            ocl::Kernel sum_tree2(sum_kernel, sum_kernel_length, "sumTree2");
            sum_tree2.compile();

            timer t;
            unsigned int initialValue = 0, sum = 0;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.copyToN(res_gpu, n);
                for (int current_n = n; current_n > 1; current_n = (current_n + workGroupSize - 1) / workGroupSize) {
                    global_work_size = (current_n + workGroupSize - 1) / workGroupSize * workGroupSize;
                    res_gpu.swap(bs_gpu);                                
                    sum_tree2.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                bs_gpu, res_gpu, current_n);
                }
                t.nextLap();
            }
            res_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            std::cout << "GPU (tree_v2): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU (tree_v2): " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
