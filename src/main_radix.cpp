#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"
#include "cl/radix_cl.h"
#include "libgpu/work_size.h"

#include <iostream>
#include <limits>
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


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
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
        std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }


    {
        ocl::Kernel radix_count(radix_kernel, radix_kernel_length, "radix_count");
        ocl::Kernel radix_sort(radix_kernel, radix_kernel_length, "radix_sort");
        ocl::Kernel transpose(radix_kernel, radix_kernel_length, "matrix_transpose");
        radix_count.compile();
        radix_sort.compile();
        transpose.compile();

        ocl::Kernel sweep_up(prefix_sum_kernel, prefix_sum_kernel_length, "sweep_up");
        ocl::Kernel sweep_down(prefix_sum_kernel, prefix_sum_kernel_length, "sweep_down");
        ocl::Kernel set_zero(prefix_sum_kernel, prefix_sum_kernel_length, "set_zero");
        sweep_up.compile();
        sweep_down.compile();
        set_zero.compile();

        unsigned int work_group_size = 16;
        unsigned int global_work_size = n;
        unsigned int bit_in_block = 4;
        unsigned int block_count = 32 / bit_in_block;
        unsigned int counter_y = 1 << bit_in_block;
        unsigned int counter_x = global_work_size / work_group_size;
        unsigned int counter_size = counter_x * counter_y;

        gpu::gpu_mem_32u as_gpu, buffer_gpu, counter_gpu, prefix_counter_gpu;
        as_gpu.resizeN(n);
        buffer_gpu.resizeN(n);
        counter_gpu.resizeN(counter_size);
        prefix_counter_gpu.resizeN(counter_size);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (int block_number = 0; block_number < block_count; ++block_number) {
                radix_count.exec(gpu::WorkSize(work_group_size, global_work_size), as_gpu, counter_gpu, block_number);

                transpose.exec(gpu::WorkSize(work_group_size, work_group_size, counter_y, counter_x), counter_gpu,
                               prefix_counter_gpu, counter_x, counter_y);

                for (int offset = 1; offset < counter_size; offset <<= 1) {
                    uint32_t globalWorkSize =
                            gpu::divup(counter_size / (offset << 1), work_group_size) * work_group_size;
                    sweep_up.exec(gpu::WorkSize(work_group_size, globalWorkSize), prefix_counter_gpu, counter_size,
                                  offset);
                }

                set_zero.exec(gpu::WorkSize(work_group_size, 1), prefix_counter_gpu, counter_size);

                for (int offset = n >> 1; offset > 0; offset >>= 1) {
                    uint32_t globalWorkSize = gpu::divup(counter_size / offset, work_group_size) * work_group_size;
                    sweep_down.exec(gpu::WorkSize(work_group_size, globalWorkSize), prefix_counter_gpu, counter_size,
                                    offset);
                }

                radix_sort.exec(gpu::WorkSize(work_group_size, global_work_size), as_gpu, prefix_counter_gpu,
                                buffer_gpu, counter_x, block_number);

                std::swap(as_gpu, buffer_gpu);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
