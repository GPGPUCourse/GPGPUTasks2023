#include "matrix_transpose.hpp"
#include "merge_sort.hpp"
#include "prefix_sum.hpp"

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


void print_data(const std::vector<uint32_t>& data, size_t n, const std::string& header, size_t new_line = 1) {
    std::cout << std::endl << header;
    for (size_t i = 0; i < n; ++i) {
        if (i % new_line == 0) {
            std::cout << std::endl;
        }
        std::cout << std::hex << data[i] << '\t';
    }
    std::cout << std::endl;
}

void print_gpu(const gpu::gpu_mem_32u &as_gpu, size_t n, const std::string& header, size_t new_line = 1) {
    std::vector<uint32_t> data(n, 0);
    as_gpu.readN(data.data(), n);
    print_data(data, n, header, new_line);
}

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
        std::cout << "CPU: " << (n / 1e6) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu, bs_gpu, cs_gpu, cs_gpu_t, res;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);
    res.resizeN(n);

    auto merge_sort = MergeSort();
    auto transpose = Transpose();
    auto prefix = PrefixSum();

    {
        ocl::Kernel count(radix_kernel, radix_kernel_length, "count");
        count.compile();
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
        radix.compile();

        constexpr auto width = 4;
        constexpr auto border = 1 << width;

        auto workGroupSize = 128;
        auto workSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

//        print_data(cpu_sorted, n, "cpu sorted", workGroupSize);

        size_t counter_rows = workSize / workGroupSize;
        size_t counter_size = border * counter_rows;
        cs_gpu.resizeN(counter_size + 1);
        cs_gpu_t.resizeN(counter_size + 1);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

//            print_gpu(as_gpu, n, "init", workGroupSize);
            t.restart(); // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (int i = 0; i < (sizeof(unsigned int) * 8) / width; ++i) {
                merge_sort.merge_sort(n, as_gpu, bs_gpu, width * i, border - 1, workGroupSize);
//                print_gpu(as_gpu, n, "sorted", workGroupSize);
                count.exec(gpu::WorkSize(workGroupSize, workSize),
                           as_gpu, cs_gpu, i);
//                print_gpu(cs_gpu, counter_size, "counted", border);

                transpose.transpose(border, counter_rows, cs_gpu, cs_gpu_t);
//                print_gpu(cs_gpu_t, counter_size, "transposed", counter_rows);

                prefix.prefix_sum(counter_size, cs_gpu);
//                print_gpu(cs_gpu, counter_size, "prefix", border);

                prefix.prefix_sum(counter_size, cs_gpu_t);
//                print_gpu(cs_gpu_t, counter_size, "prefix t", counter_rows);

                radix.exec(gpu::WorkSize(workGroupSize, workSize),
                           as_gpu, i, cs_gpu, cs_gpu_t, res);
//                print_gpu(res, n, "res", workGroupSize);

                as_gpu.swap(res);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1e6) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results! At: " + std::to_string(i));
    }

    return 0;
}
