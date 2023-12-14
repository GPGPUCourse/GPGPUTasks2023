#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/matrix_transpose_cl.h"
#include "cl/merge_cl.h"
#include "cl/prefix_sum_cl.h"
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

ocl::Kernel get_kernel(const char *file, size_t file_len, std::string kernel_name, std::string defines = "") {
    ocl::Kernel kernel(file, file_len, kernel_name, defines);
    kernel.compile();
    return kernel;
}

gpu::gpu_mem_32u get_mem(size_t size) {
    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(size);
    return as_gpu;
}

void print_gpu_mem(gpu::gpu_mem_32u as_gpu, uint n) {
    std::vector<uint> res(n, 0);
    as_gpu.readN(res.data(), n);

    for (auto a : res)
        std::cout << a << ' ';
    std::cout << '\n';
}

void print_gpu_mem(gpu::gpu_mem_32u as_gpu, uint n, uint m) {
    std::vector<uint> res(n * m, 0);
    as_gpu.readN(res.data(), n * m);

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            std::cout << res[i * m + j] << ' ';
        std::cout << '\n';
    }
}


uint n = 32 * 1024 * 1024;
uint k = 4;
uint wg_size = 256;


struct PrefixSum {

    void operator()(gpu::gpu_mem_32u &as_gpu, gpu::gpu_mem_32u sums_gpu, uint n, gpu::gpu_mem_32u &temp_gpu) {
        uint wg_size = std::min(n, 128u);
        for (int sum_len = 1; sum_len <= n; sum_len <<= 1) {
            if (sum_len != 1) {
                std::swap(as_gpu, temp_gpu);
                uint wg_size = std::min(n / sum_len, 128u);
                reduce_kernel.exec(gpu::WorkSize(wg_size, n / sum_len), temp_gpu, as_gpu);
            }
            sum_if_need_kernel.exec(gpu::WorkSize(wg_size, n), as_gpu, sums_gpu, sum_len);
        }
    }
    ocl::Kernel reduce_kernel = get_kernel(prefix_sum_kernel, prefix_sum_kernel_length, "reduce");
    ocl::Kernel sum_if_need_kernel = get_kernel(prefix_sum_kernel, prefix_sum_kernel_length, "sum_if_need");
};

struct MergeSort {

    void operator()(gpu::gpu_mem_32u &as_gpu, uint merge_size, uint shift, uint k, gpu::gpu_mem_32u &temp_gpu) {
        unsigned int workGroupSize = std::min(n, 128u);
        unsigned int global_work_size = n;
        {
            unsigned int cur_merge_size = 1;
            while (cur_merge_size < merge_size) {
                merge_sort_kernel.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, temp_gpu, n,
                                       cur_merge_size, shift, k);
                cur_merge_size *= 2;
                std::swap(as_gpu, temp_gpu);
            }
        }
    }

    ocl::Kernel merge_sort_kernel = get_kernel(merge_kernel, merge_kernel_length, "merge");
};

struct MatrixTranspose {

    void operator()(gpu::gpu_mem_32u m_gpu, gpu::gpu_mem_32u res_gpu, uint height, uint width) {
        uint wg_x = 16;
        uint wg_y = 16;
        gpu::WorkSize ws(wg_x, wg_y, width, height);
        transpose_kernel.exec(ws, m_gpu, res_gpu, height, width);
    }

    ocl::Kernel transpose_kernel =
            get_kernel(matrix_transpose_kernel, matrix_transpose_kernel_length, "matrix_transpose");
};

struct RadixSort {


    void operator()(ocl::Kernel calc_counters_kernel, ocl::Kernel radix_kernel, MatrixTranspose matrix_transpose,
                    MergeSort merge_sort, PrefixSum prefix_sum, gpu::gpu_mem_32u &as_gpu, uint k) {
        gpu::WorkSize ws(wg_size, n);
        uint wg_count = n / wg_size;
        zeros_gpu.writeN(n_zeros.data(), counters_size);

        for (int shift = 0; shift < sizeof(uint) * 8; shift += k) {
            // std::cout << "As:\n";
            // print_gpu_mem(as_gpu, n);

            merge_sort(as_gpu, wg_size, shift, k, temp_gpu_n);
            // std::cout << "As after merge:\n";
            // print_gpu_mem(as_gpu, n);
            zeros_gpu.copyToN(counters_gpu, counters_size);
            calc_counters_kernel.exec(ws, as_gpu, shift, counters_gpu);
            // std::cout << "Counters:\n";
            // print_gpu_mem(counters_gpu, wg_count, wg_size);

            //counters with sizes wg_count x (1 << k)
            matrix_transpose(counters_gpu, counters_t_gpu, wg_count, 1 << k);
            // std::cout << "Counters T:\n";
            // print_gpu_mem(counters_t_gpu, wg_size, wg_count);

            zeros_gpu.copyToN(sums_gpu, counters_size);
            prefix_sum(counters_gpu, sums_gpu, counters_size, temp_gpu);
            // std::cout << "Sums:\n";
            // print_gpu_mem(sums_gpu, wg_count, wg_size);

            zeros_gpu.copyToN(sums_t_gpu, counters_size);
            prefix_sum(counters_t_gpu, sums_t_gpu, counters_size, temp_gpu);
            // std::cout << "Sums T:\n";
            // print_gpu_mem(sums_t_gpu, wg_size, wg_count);


            radix_kernel.exec(ws, as_gpu, shift, sums_gpu, sums_t_gpu, temp_gpu_n);
            // std::cout << "Radix:\n";
            // print_gpu_mem(temp_gpu, n);

            std::swap(as_gpu, temp_gpu_n);
        }
    }
    uint counters_size = n / wg_size * (1 << k);
    std::vector<uint> n_zeros = std::vector<uint>(counters_size, 0);
    gpu::gpu_mem_32u counters_gpu = get_mem(counters_size);
    gpu::gpu_mem_32u temp_gpu_n = get_mem(n);
    gpu::gpu_mem_32u temp_gpu = get_mem(counters_size);
    gpu::gpu_mem_32u zeros_gpu = get_mem(counters_size);
    gpu::gpu_mem_32u counters_t_gpu = get_mem(counters_size);
    gpu::gpu_mem_32u sums_gpu = get_mem(counters_size);
    gpu::gpu_mem_32u sums_t_gpu = get_mem(counters_size);
    gpu::gpu_mem_32u res_gpu = get_mem(n);
};


void test_radix() {
    std::vector<uint> as(n, 0);
    std::vector<uint> res(n, 0);
    FastRandom r;
    uint k = 2;
    for (uint i = 0; i < n; ++i)
        as[i] = r.next(0, 100);

    auto defines = "-DWG_SIZE=" + to_string(1 << k);
    auto calc_counters = get_kernel(radix_kernel, radix_kernel_length, "calc_counters", defines);
    auto radix = get_kernel(radix_kernel, radix_kernel_length, "radix_sort", defines);
    auto as_gpu = get_mem(n);
    RadixSort radix_sort;
    MatrixTranspose matrix_transpose;
    MergeSort merge_sort;
    PrefixSum prefix_sum;

    as_gpu.writeN(as.data(), n);
    radix_sort(calc_counters, radix, matrix_transpose, merge_sort, prefix_sum, as_gpu, k);
    std::cout << "Radix:\n";
    print_gpu_mem(as_gpu, n);
}

void test_calc_counters() {
    uint n = 128;
    std::vector<uint> as(n, 0);
    FastRandom r(n);
    for (uint i = 0; i < n; ++i)
        as[i] = r.next(0, 127);
    uint k = 4;
    auto calc_counters =
            get_kernel(radix_kernel, radix_kernel_length, "calc_counters", "-DWG_SIZE=" + to_string(1 << k));
    gpu::WorkSize ws(1 << k, n);
    auto as_gpu = get_mem(n);
    auto res_gpu = get_mem(n);
    as_gpu.writeN(as.data(), n);
    for (int shift = 0; shift < sizeof(uint) * 8; shift += k) {
        calc_counters.exec(ws, as_gpu, shift, k, res_gpu);
        std::vector<uint> res(n, 0);
        res_gpu.readN(res.data(), n);
        for (auto a : as)
            std::cout << a << ' ';
        std::cout << '\n';
        for (auto r : res)
            std::cout << r << ' ';
        std::cout << "\n\n";
        res = std::vector<uint>(n, 0);
        res_gpu.writeN(res.data(), n);
    }
}


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();
    //test_radix();
    // return 0;

    int benchmarkingIters = 10;
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

        auto defines = "-DWG_SIZE=" + to_string(wg_size) + " -DBITS_COUNT=" + to_string(k);
        auto calc_counters = get_kernel(radix_kernel, radix_kernel_length, "calc_counters", defines);
        auto radix = get_kernel(radix_kernel, radix_kernel_length, "radix_sort", defines);
        auto as_gpu = get_mem(n);
        RadixSort radix_sort;
        MatrixTranspose matrix_transpose;
        MergeSort merge_sort;
        PrefixSum prefix_sum;

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            radix_sort(calc_counters, radix, matrix_transpose, merge_sort, prefix_sum, as_gpu, k);
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

    {
        auto as_gpu = get_mem(n);
        auto temp_gpu = get_mem(n);
        MergeSort merge_sort;

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            merge_sort(as_gpu, n, 0, sizeof(uint) * 8, temp_gpu);
            t.nextLap();
        }
        std::cout << "GPU (merge): " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU (merge): " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
