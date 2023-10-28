#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"

#include <iostream>
#include <sstream>
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

    constexpr int benchmarkingIters = 1; // 10;
    constexpr cl_uint n = 4 * 1024 * 1024; // 32 * 1024 * 1024;
    std::vector<cl_uint> as(n, 0);
    FastRandom r(n);
    for (cl_uint i = 0; i < n; ++i) {
        as[i] = static_cast<cl_uint>(r.next(0, std::numeric_limits<int>::max()));
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // for (cl_uint i = 0; i < n; ++i) {
    //     std::cout << "as[" << i << "] = " << as[i] << "\n";
    // }

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << 1e-6 * n / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        constexpr cl_uint BIT_SIZE = 32;
        constexpr cl_uint STEP_BITS = 1; // With 8 we already run out of local memory
        constexpr cl_uint N_STEPS = BIT_SIZE / STEP_BITS;
        constexpr cl_uint RADIX_BASE = 1 << STEP_BITS;
        constexpr cl_uint RADIX_MASK = RADIX_BASE - 1;
        constexpr cl_uint WORK_GROUP_SIZE = 64;
        constexpr cl_uint N_WORK_GROUPS = (n + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
        static_assert(BIT_SIZE % STEP_BITS == 0);

        constexpr cl_uint N_OFFSETS = RADIX_BASE * N_WORK_GROUPS;

        const gpu::WorkSize ws_n(WORK_GROUP_SIZE, n);
        const gpu::WorkSize ws_cumsum(WORK_GROUP_SIZE, N_OFFSETS);

        gpu::gpu_mem_32u gpu_src;
        gpu::gpu_mem_32u gpu_dst;
        gpu::gpu_mem_32u gpu_off_src;
        gpu::gpu_mem_32u gpu_off_dst;
        gpu_src.resizeN(n);
        gpu_dst.resizeN(n);
        gpu_off_src.resizeN(N_OFFSETS);
        gpu_off_dst.resizeN(N_OFFSETS);

        std::ostringstream defines;
        defines << "-DRADIX_BASE=" << RADIX_BASE << " -DRADIX_MASK=" << RADIX_MASK
                << " -DWORK_GROUP_SIZE=" << WORK_GROUP_SIZE;

        ocl::Kernel k_count(radix_kernel, radix_kernel_length, "radix_count_local_t", defines.str());
        ocl::Kernel k_reorder(radix_kernel, radix_kernel_length, "radix_reorder", defines.str());
        ocl::Kernel k_cumsum(radix_kernel, radix_kernel_length, "cumsum_naive", defines.str());
        k_count.compile();
        k_reorder.compile();
        k_cumsum.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            gpu_src.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (cl_uint shift = 0; shift < BIT_SIZE; shift += STEP_BITS) {
                k_count.exec(ws_n, gpu_src, gpu_off_src, n, shift);

                for (cl_uint step = 1; step < N_OFFSETS; step *= 2) {
                    k_cumsum.exec(ws_cumsum, gpu_off_src, gpu_off_dst, N_OFFSETS, step);
                    std::swap(gpu_off_src, gpu_off_dst);
                }

                k_reorder.exec(ws_n, gpu_src, gpu_dst, gpu_off_src, n, shift);
                std::swap(gpu_src, gpu_dst);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << 1e-6 * n / t.lapAvg() << " millions/s" << std::endl;

        gpu_src.readN(as.data(), n);
    }

    // for (cl_uint i = 0; i < n; ++i) {
    //     std::cout << "as[" << i << "] = " << as[i] << "\n";
    // }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    // */

    return 0;
}
