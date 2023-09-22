#include "cl/sum_cl.h"

#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"

#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>


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
    constexpr int benchmarkingIters = 20;
    constexpr unsigned int n = 200*1000*1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);

    unsigned int reference_sum = 0;
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    std::cout << "Generated " << n / 1000 / 1000 << " million numbers" << std::endl << std::endl;

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
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl << std::endl;
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
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl << std::endl;
    }

    constexpr unsigned int workGroupSize = 128;
    constexpr unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

    {
        // 3.2.1 Суммирование с глобальным атомарным добавлением (просто как бейзлайн)
        std::cout << "3.2.1  GPU baseline (atomic_add)" << std::endl;

        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu, sum_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);

        sum_gpu.resizeN(1);

        ocl::Kernel kernel_sum1(sum_kernel, sum_kernel_length, "kernel_sum1");
        kernel_sum1.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            sum_gpu.write(&sum, sizeof(unsigned int));

            kernel_sum1.exec(gpu::WorkSize(workGroupSize, global_work_size)
                      , as_gpu
                      , sum_gpu
                      , n);

            sum_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl << std::endl;
    }

    {
        // 3.2.2 Суммирование с циклом
        std::cout << "3.2.2  GPU sum with cycle" << std::endl;
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu, sum_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);

        sum_gpu.resizeN(1);

        ocl::Kernel kernel_sum2(sum_kernel, sum_kernel_length, "kernel_sum2");
        kernel_sum2.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            sum_gpu.write(&sum, sizeof(unsigned int));

            kernel_sum2.exec(gpu::WorkSize(workGroupSize, global_work_size / workGroupSize)
                    , as_gpu
                    , sum_gpu
                    , workGroupSize
                    , n);

            sum_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl << std::endl;
    }

    {
        // 3.2.3 Суммирование с циклом и coalesced доступом (интересно сравнение по скорости с не-coalesced версией)
        std::cout << "3.2.3  GPU sum with cycle and coalesced access" << std::endl;
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu, sum_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);

        sum_gpu.resizeN(1);

        ocl::Kernel kernel_sum3(sum_kernel, sum_kernel_length, "kernel_sum3");
        kernel_sum3.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            sum_gpu.write(&sum, sizeof(unsigned int));

            kernel_sum3.exec(gpu::WorkSize(workGroupSize, global_work_size / workGroupSize)
                    , as_gpu
                    , sum_gpu
                    , workGroupSize
                    , n);

            sum_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl << std::endl;
    }

    {
        //3.2.4 Суммирование с локальной памятью и главным потоком
        std::cout << "3.2.4  GPU sum with local memory and main thread" << std::endl;
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu, sum_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);

        sum_gpu.resizeN(1);

        ocl::Kernel kernel_sum4(sum_kernel, sum_kernel_length, "kernel_sum4");
        kernel_sum4.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            sum_gpu.write(&sum, sizeof(unsigned int));

            kernel_sum4.exec(gpu::WorkSize(workGroupSize, global_work_size)
                    , as_gpu
                    , sum_gpu
                    , n);

            sum_gpu.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl << std::endl;
    }

    {
        //3.2.5 Суммирование с деревом
        std::cout << "3.2.5  GPU sum with tree" << std::endl;
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu, sum_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);

        ocl::Kernel kernel_sum5(sum_kernel, sum_kernel_length, "kernel_sum5");
        kernel_sum5.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int current_size = n;
            std::vector<unsigned int> sum(current_size / workGroupSize + (current_size % workGroupSize ? 1 : 0), 0);
            sum_gpu.resizeN(sum.size());

            kernel_sum5.exec(gpu::WorkSize(workGroupSize, current_size)
                    , as_gpu
                    , sum_gpu
                    , current_size);

            sum_gpu.readN(sum.data(), sum.size());
            unsigned int result_sum = 0;
            for(const auto& v: sum) {
                result_sum += v;
            }
            EXPECT_THE_SAME(reference_sum, result_sum, "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl << std::endl;
    }

    {
        //3.2.5* Суммирование с деревом рекурсивно
        std::cout << "3.2.5*  GPU recursive sum with tree" << std::endl;
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu, sum_gpu;

        ocl::Kernel kernel_sum5(sum_kernel, sum_kernel_length, "kernel_sum5");
        kernel_sum5.compile();

        as_gpu.resizeN(n);
        sum_gpu.resizeN(n / workGroupSize + (n % workGroupSize ? 1 : 0));

        constexpr unsigned int threshold = 2 * workGroupSize;

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {

            t.stop();
            as_gpu.writeN(as.data(), n);
            t.start();

            unsigned int current_size = n;

            while(current_size >= threshold) {
                kernel_sum5.exec(gpu::WorkSize(workGroupSize, current_size)
                        , as_gpu
                        , sum_gpu
                        , current_size);
                current_size = current_size / workGroupSize + (current_size % workGroupSize ? 1 : 0);
                sum_gpu.copyToN(as_gpu, current_size);
            }

            std::vector<unsigned int> sum(workGroupSize, 0);
            sum_gpu.readN(sum.data(), current_size);
            for(int i = 1; i < current_size; ++i) {
                sum[0] += sum[i];
            }
            EXPECT_THE_SAME(reference_sum, sum[0], "GPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl << std::endl;
    }
}
