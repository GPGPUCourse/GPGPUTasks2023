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
#include <string_view>
#include <vector>

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

constexpr int BENCHMARKING_ITERS = 1;// 10;
constexpr cl_uint N = 32 * 1024 * 1024;
constexpr cl_uint WORK_GROUP_SIZE = 64;

std::vector<cl_uint> as(N, 0);

template<typename T>
void dbg_vec(std::string_view name, const std::vector<T> &v) {
    for (size_t i = 0; i < v.size(); ++i) {
        std::cerr << name << "[" << i << "] = " << v[i] << "\n";
    }
}

void radix_local(const cl_uint STEP_BITS, bool local, bool transposed) {
    constexpr cl_uint BIT_SIZE = 32;
    // constexpr cl_uint STEP_BITS = 2;
    // static_assert(BIT_SIZE % STEP_BITS == 0);
    const cl_uint N_STEPS = BIT_SIZE / STEP_BITS;
    const cl_uint RADIX_BASE = 1 << STEP_BITS;
    const cl_uint RADIX_MASK = RADIX_BASE - 1;
    const cl_uint N_WORK_GROUPS = (N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
    const cl_uint N_OFFSETS = RADIX_BASE * N_WORK_GROUPS;
    const cl_uint TILE_SIZE = std::min(RADIX_BASE, cl_uint(32));

    const gpu::WorkSize ws_count(WORK_GROUP_SIZE, N);
    const gpu::WorkSize ws_matrix(TILE_SIZE, RADIX_BASE, TILE_SIZE, N_WORK_GROUPS);
    const gpu::WorkSize ws_offset(WORK_GROUP_SIZE, N_OFFSETS);

    gpu::gpu_mem_32u gpu_src;
    gpu::gpu_mem_32u gpu_dst;
    gpu::gpu_mem_32u gpu_off_src;
    gpu::gpu_mem_32u gpu_off_dst;
    gpu_src.resizeN(N);
    gpu_dst.resizeN(N);
    gpu_off_src.resizeN(N_OFFSETS);
    gpu_off_dst.resizeN(N_OFFSETS);

    std::ostringstream defines;
    defines << "-DRADIX_BASE=" << RADIX_BASE << " -DRADIX_MASK=" << RADIX_MASK
            << " -DWORK_GROUP_SIZE=" << WORK_GROUP_SIZE << " -DTILE_SIZE=" << TILE_SIZE;

    std::ostringstream k_count_name;
    k_count_name << "radix_count";
    if (local) {
        k_count_name << "_local";
    }
    if (transposed) {
        k_count_name << "_t";
    }

    ocl::Kernel k_reset(radix_kernel, radix_kernel_length, "fill_zero", defines.str());
    ocl::Kernel k_count(radix_kernel, radix_kernel_length, k_count_name.str(), defines.str());
    ocl::Kernel k_transpose(radix_kernel, radix_kernel_length, "matrix_transpose", defines.str());
    ocl::Kernel k_reorder(radix_kernel, radix_kernel_length, "radix_reorder", defines.str());
    ocl::Kernel k_cumsum(radix_kernel, radix_kernel_length, "cumsum_naive", defines.str());
    k_count.compile();
    k_reorder.compile();
    k_cumsum.compile();

    timer t;
    for (int iter = 0; iter < BENCHMARKING_ITERS; ++iter) {
        gpu_src.writeN(as.data(), N);

        t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

        for (cl_uint shift = 0; shift < BIT_SIZE; shift += STEP_BITS) {
            if (!local) {
                k_reset.exec(ws_offset, gpu_off_src, N_OFFSETS);
            }

            k_count.exec(ws_count, gpu_src, gpu_off_src, N, shift);

            if (!transposed) {
                k_transpose.exec(ws_matrix, gpu_off_src, gpu_off_dst, RADIX_BASE, N_WORK_GROUPS);
                std::swap(gpu_off_src, gpu_off_dst);
            }

            for (cl_uint step = 1; step < N_OFFSETS; step *= 2) {
                k_cumsum.exec(ws_offset, gpu_off_src, gpu_off_dst, N_OFFSETS, step);
                std::swap(gpu_off_src, gpu_off_dst);
            }

            k_reorder.exec(ws_count, gpu_src, gpu_dst, gpu_off_src, N, shift);
            std::swap(gpu_src, gpu_dst);
        }

        t.nextLap();
    }

    std::ostringstream run_name;
    run_name << "GPU";
    if (local) {
        run_name << " local";
    }
    if (transposed) {
        run_name << " transposed";
    }
    run_name << " step=" << STEP_BITS << ": ";

    std::cout << "\n";
    std::cout << run_name.str() << t.lapAvg() << "+-" << t.lapStd() << " s\n";
    std::cout << run_name.str() << 1e-6 * N / t.lapAvg() << " millions/s\n";

    gpu_src.readN(as.data(), N);
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    FastRandom r(N);
    for (cl_uint i = 0; i < N; ++i) {
        as[i] = static_cast<cl_uint>(r.next(0, std::numeric_limits<int>::max()));
    }
    std::cout << "Data generated for n=" << N << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < BENCHMARKING_ITERS; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << 1e-6 * N / t.lapAvg() << " millions/s" << std::endl;
    }

    // 8 уже слишком много, 256 * 32 уже не влезает в локальную память,
    // а делать work group меньше 32 нет смысла.
    for (cl_uint STEP_BITS = 1; STEP_BITS < 8; STEP_BITS *= 2) {
        radix_local(STEP_BITS, false, false);
        radix_local(STEP_BITS, false, true);
        radix_local(STEP_BITS, true, false);
        radix_local(STEP_BITS, true, true);
    }

    // dbg_vec("cpu_sorted", cpu_sorted);
    // dbg_vec("as", as);

    // Проверяем корректность результатов
    for (int i = 0; i < N; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    // */

    return 0;
}
