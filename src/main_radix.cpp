#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <cassert>
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

template<typename T>
void debug(const std::string &label, const std::vector<T> &a) {
    std::cout << label << std::endl;
    std::cout << "[";
    for (int i = 0; i < a.size(); ++i) {
        if (i) {
            std::cout << ", " << std::endl;
        }
        std::cout << label << "[" << i << "] = " << a[i];
    }
    std::cout << "]" << std::endl;
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
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        const int B_LOG = 4;
        const int B = 1 << B_LOG;
        const int TILE_SIZE = 8;// | B, | n / WG_SIZE
        const int WG_SIZE = 128;

        assert((n / WG_SIZE) % TILE_SIZE == 0);
        assert(B % TILE_SIZE == 0);

        const std::string defines = "-D TILE_SIZE=" + std::to_string(TILE_SIZE) + " -D B=" + std::to_string(B) +
                                    " -D WG_SIZE=" + std::to_string(WG_SIZE);
        auto kernel = [&defines](const std::string &label) {
            ocl::Kernel kernel(radix_kernel, radix_kernel_length, label, defines);
            kernel.compile();
            return kernel;
        };
        auto allocN = [](unsigned n) {
            gpu::gpu_mem_32u mem;
            mem.resizeN(n);
            return mem;
        };

        auto build_count = kernel("build_count");
        auto transpose = kernel("transpose");
        auto prefix_sum_sparse = kernel("prefix_sum_sparse");
        auto prefix_sum_supplement = kernel("prefix_sum_supplement");
        auto merge = kernel("merge");
        auto radix_sort = kernel("radix_sort");
        auto dummy_prefix_sums = kernel("dummy_prefix_sums");

        const unsigned counter_sz = n / WG_SIZE * B;
        const unsigned bufSize = std::max(n, counter_sz);

        auto as_gpu = allocN(bufSize);
        auto counters_gpu = allocN(bufSize);
        auto temp_gpu = allocN(bufSize);
        auto temp2_gpu = allocN(bufSize);
        auto pf_sums_gpu = allocN(bufSize);

        auto counters_pf_gpu = allocN(bufSize);

        auto n_ws = gpu::WorkSize(WG_SIZE, n);
        auto counter_ws = gpu::WorkSize(WG_SIZE, counter_sz);
        auto transpose_ws = gpu::WorkSize(TILE_SIZE, TILE_SIZE, B, n / WG_SIZE);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (int offset = 0; offset < sizeof(unsigned) * 8; offset += B_LOG) {
                // Presort blocks.
                for (int sorted = 1; sorted < WG_SIZE; sorted *= 2) {
                    merge.exec(n_ws, as_gpu, as_gpu, sorted, offset);
                }

                // Build counters.
                build_count.exec(n_ws, as_gpu, offset, counters_gpu);

                // Calc dummy prefix sums, hope it will run faster than regular pf.
                dummy_prefix_sums.exec(counter_ws, counters_gpu, counters_pf_gpu, B);

                // Transpose.
                transpose.exec(transpose_ws, counters_gpu, temp_gpu, n / WG_SIZE, B);

                // Calculate prefix sums.
                for (int csz = 1; csz <= counter_sz; csz *= 2) {
                    prefix_sum_supplement.exec(counter_ws, temp_gpu, pf_sums_gpu, csz);
                    prefix_sum_sparse.exec(counter_ws, temp_gpu, temp2_gpu, csz, counter_sz);
                    temp_gpu.swap(temp2_gpu);
                }

                // Radix.
                radix_sort.exec(n_ws, as_gpu, temp_gpu, pf_sums_gpu, n / WG_SIZE, offset, counters_pf_gpu);
                as_gpu.swap(temp_gpu);
            }

            t.nextLap();
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
