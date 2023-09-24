#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/timer.h>

#include "cl/sum_cl.h"

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void sum_gpu_bench(const std::string &kernel_name, const gpu::gpu_mem_32u &gpu_data,
                   unsigned int reference_sum, unsigned int n, unsigned wg_size, unsigned grid_size,
                   int benchmark_iters = 10) {
    ocl::Kernel sum_gpu(sum_kernel, sum_kernel_length, kernel_name);
    sum_gpu.compile();
    timer t;
    for (int iter = 0; iter < benchmark_iters; ++iter) {
        unsigned int sum = 0;
        gpu::gpu_mem_32u gpu_sum;
        gpu_sum.resizeN(1);
        gpu_sum.writeN(&sum, 1);

        sum_gpu.exec(gpu::WorkSize(wg_size, grid_size),
                     gpu_sum, gpu_data, n);

        gpu_sum.readN(&sum, 1);
        EXPECT_THE_SAME(reference_sum, sum, kernel_name + " result should be consistent!");
        t.nextLap();
    }
    std::cout << kernel_name << ":     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << kernel_name << ":     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
}

int main(int argc, char **argv) {
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
//        as[i] = (unsigned int)1;
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
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
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
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    { ;
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        // 4 x wave32 on Navi3

        unsigned wg_size = 32;
        // on Navi3 doesn't return the number of CUs, but the number of SAs
        unsigned cu_count = device.compute_units * 2;
        unsigned wave_slots_per_simd = 16;

        unsigned grid_size = wg_size * cu_count * wave_slots_per_simd;
        gpu::gpu_mem_32u gpu_data;
        gpu_data.resizeN(n);
        gpu_data.writeN(as.data(), n);
        sum_gpu_bench("sum_atomic", gpu_data, reference_sum, n, wg_size, grid_size);
        sum_gpu_bench("sum_atomic_iter", gpu_data, reference_sum, n, wg_size, grid_size);
        sum_gpu_bench("sum_atomic_coalesce", gpu_data, reference_sum, n, wg_size, grid_size);

        wg_size = 128;
        grid_size = wg_size * cu_count * wave_slots_per_simd;
        sum_gpu_bench("sum_local", gpu_data, reference_sum, n, wg_size, grid_size);
        wg_size = 128;
        grid_size = wg_size * cu_count * wave_slots_per_simd;
        sum_gpu_bench("sum_tree", gpu_data, reference_sum, n, wg_size, grid_size);
    }
}
