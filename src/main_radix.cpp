#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

#define MOVE (1 << 4)

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 1;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    const int WORK_GROUP_SIZE = 128;

    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u bs_gpu;
    gpu::gpu_mem_32u cs_gpu;
    gpu::gpu_mem_32u ds_gpu;
    gpu::gpu_mem_32u es_gpu;

    unsigned int siz = MOVE * n / WORK_GROUP_SIZE;

    as_gpu.resizeN(n);
    bs_gpu.resizeN(siz);
    cs_gpu.resizeN(siz);
    es_gpu.resizeN(siz);
    ds_gpu.resizeN(n);

    std::vector<unsigned int> bs(siz);
    {
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        // new kernels
        ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
        ocl::Kernel transpose(radix_kernel, radix_kernel_length, "transpose");
        ocl::Kernel prefix(radix_kernel, radix_kernel_length, "prefix");
        ocl::Kernel reduce(radix_kernel, radix_kernel_length, "reduce");
        ocl::Kernel merge(radix_kernel, radix_kernel_length, "merge");
        ocl::Kernel zero(radix_kernel, radix_kernel_length, "zero");
        ocl::Kernel move_(radix_kernel, radix_kernel_length, "move");
        ocl::Kernel local_prefix(radix_kernel, radix_kernel_length, "local_prefix");

        count.compile();
        transpose.compile();
        prefix.compile();
        reduce.compile();
        merge.compile();
        zero.compile();
        move_.compile();
        local_prefix.compile();

        const int WORK_GROUP_X = 16, WORK_GROUP_Y = 16;

        auto work_size = [](int n) {
            unsigned int workGroupSize = WORK_GROUP_SIZE;
            unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
            return gpu::WorkSize(workGroupSize, global_work_size);
        };
        auto work_size_2 = [&WORK_GROUP_X, &WORK_GROUP_Y](int n, int m) {
            unsigned int workSizeX = (n + WORK_GROUP_X - 1) / WORK_GROUP_X * WORK_GROUP_X;
            unsigned int workSizeY = (m + WORK_GROUP_Y - 1) / WORK_GROUP_Y * WORK_GROUP_Y;
            return gpu::WorkSize(WORK_GROUP_X, WORK_GROUP_Y, workSizeX, workSizeY);
        };

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            bs_gpu.writeN(bs.data(), siz);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (unsigned int stride = 0; stride < 32; stride += 4) {
                merge.exec(work_size(n), as_gpu, ds_gpu, stride);
                count.exec(work_size(n), ds_gpu, bs_gpu, stride);
                move_.exec(work_size(siz), bs_gpu, es_gpu);

                transpose.exec(work_size_2(MOVE, n / WORK_GROUP_SIZE), bs_gpu, cs_gpu, n / WORK_GROUP_SIZE, MOVE);

                for (unsigned int step = 1; step <= siz / 2; step <<= 1) {
                    unsigned int groups = (((siz & (~step)) - (siz & (step - 1))) >> 1) + (siz & (step - 1));

                    prefix.exec(work_size(groups), cs_gpu, bs_gpu, step, siz);
                    if (step < siz / 2)
                        reduce.exec(work_size(siz / step / 2), cs_gpu, siz, step);
                }

                local_prefix.exec(gpu::WorkSize(MOVE, MOVE * n / WORK_GROUP_SIZE), es_gpu);
                radix.exec(work_size(n), ds_gpu, as_gpu, bs_gpu, es_gpu, n, stride);
                zero.exec(work_size(siz), bs_gpu);
            }

            t.nextLap();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
