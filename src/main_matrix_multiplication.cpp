#include "../libs/utils/libutils/misc.h"
#include "../libs/utils/libutils/timer.h"
#include "../libs/utils/libutils/fast_random.h"
#include "../libs/gpu/libgpu/context.h"
#include "../libs/gpu/libgpu/shared_device_buffer.h"
#include "../libs/gpu/libgpu/work_size.h"
#include "../libs/gpu/libgpu/device.h"

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>

constexpr int benchmarkingIters = 10; // TODO пока тестируетесь удобно выставить единицу
constexpr unsigned int M = 1024;
constexpr unsigned int K = 1024;
constexpr unsigned int N = 1024;
constexpr size_t gflops =
        ((size_t) M * K * N * 2) / (1000 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения

void test_kernel(std::vector<float> &as, std::vector<float> &bs, const std::string kernel_name, const float *expected,
                 int x_scale = 1, int y_scale = 1) {

    gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;
    as_gpu.resizeN(M * K);
    bs_gpu.resizeN(K * N);
    cs_gpu.resizeN(M * N);

    as_gpu.writeN(as.data(), M * K);
    bs_gpu.writeN(bs.data(), K * N);
    ocl::Kernel matrix_multiplication_kernel(matrix_multiplication, matrix_multiplication_length,
                                             kernel_name);
    matrix_multiplication_kernel.compile();

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            // TODO
            unsigned int work_group_size = 16;
            unsigned int global_work_size_width =
                    (((N + x_scale - 1) / x_scale) + work_group_size - 1) / work_group_size * work_group_size;
            unsigned int global_work_size_height =
                    (((M + y_scale - 1) / y_scale) + work_group_size - 1) / work_group_size * work_group_size;
            matrix_multiplication_kernel.exec(
                    gpu::WorkSize(work_group_size, work_group_size, global_work_size_width, global_work_size_height),
                    as_gpu, bs_gpu, cs_gpu, M, K, N);

            t.nextLap();
        }
        std::cout << "GPU: (" << kernel_name << ") " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: (" << kernel_name << ") " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }
    std::vector<float> cs(M * N, 0);
    cs_gpu.readN(cs.data(), M * N);


    // Проверяем корректность результатов
    double diff_sum = 0;
    for (int i = 0; i < M * N; ++i) {
        double a = cs[i];
        double b = expected[i];
        if (a != 0.0 || b != 0.0) {
            double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
            diff_sum += diff;
        }
    }

//    double diff_avg = diff_sum / (M * N);
//    std::cout << "Average difference: " << diff_avg * 100.0 << "%" << std::endl;
//    if (diff_avg > 0.01) {
//        std::cerr << "Too big difference!" << std::endl;
//        return;
//    }
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();


    std::vector<float> as(M * K, 0);
    std::vector<float> bs(K * N, 0);
    std::vector<float> cs(M * N, 0);

    FastRandom r(M + K + N);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    for (unsigned int i = 0; i < bs.size(); ++i) {
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for M=" << M << ", K=" << K << ", N=" << N << std::endl;

//    {
//        timer t;
//        for (int iter = 0; iter < benchmarkingIters; ++iter) {
//            for (int j = 0; j < M; ++j) {
//                for (int i = 0; i < N; ++i) {
//                    float sum = 0.0f;
//                    for (int k = 0; k < K; ++k) {
//                        sum += as.data()[j * K + k] * bs.data()[k * N + i];
//                    }
//                    cs.data()[j * N + i] = sum;
//                }
//            }
//            t.nextLap();
//        }
//        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
//        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
//    }

    const std::vector<float> cs_cpu_reference = cs;

    test_kernel(as, bs, "matrix_multiplication_naive", cs_cpu_reference.data());
    test_kernel(as, bs, "matrix_multiplication_local_memes", cs_cpu_reference.data());
    test_kernel(as, bs, "matrix_multiplication_task3_rows", cs_cpu_reference.data(), 16);
    test_kernel(as, bs, "matrix_multiplication_task3_columns", cs_cpu_reference.data(), 1, 16);
    test_kernel(as, bs, "matrix_multiplication_task3_columns_no_memes", cs_cpu_reference.data(), 1, 16);
    test_kernel(as, bs, "matrix_multiplication_task3_rows_no_memes", cs_cpu_reference.data(), 16);

    return 0;
}
