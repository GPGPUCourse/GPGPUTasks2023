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


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 1;//10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int)r.next(0, std::numeric_limits<int>::max());
        //std::cout << as[i] << ' ';
    }
    //std::cout << std::endl;
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
        std::cout << "CPU: " << (n / 1000. / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);

    {
        const unsigned int BITS_IN_DIGIT = 2;
        const unsigned int TRANSPOSE_TILE_M = 16;
        const unsigned int WG_SIZE = 16;//128;

        const unsigned int N_DIGITS = sizeof(unsigned int) * 8 / BITS_IN_DIGIT;
        const unsigned int N_WG = gpu::divup(n, WG_SIZE);
        const unsigned int CNT_SIZE = 1 << BITS_IN_DIGIT;
        const unsigned int CNT_TOTAL = N_WG * CNT_SIZE;

        std::cout << "n_digits = " << N_DIGITS << "; n_wg = " << N_WG << std::endl;
        std::cout << "counters size = " << N_WG * CNT_SIZE << std::endl;

        std::string param_string = "-DBITS_IN_DIGIT=" + std::to_string(BITS_IN_DIGIT) +
                                   " -DTILE_SIZE_M=" + std::to_string(TRANSPOSE_TILE_M) +
                                   " -DTILE_SIZE_K=" + std::to_string(N_DIGITS) +
                                   " -DWG_SIZE=" + std::to_string(WG_SIZE);
        ocl::Kernel small_sort(radix_kernel, radix_kernel_length, "small_sort", param_string);
        small_sort.compile();
        ocl::Kernel counters(radix_kernel, radix_kernel_length, "counters", param_string);
        counters.compile();
        ocl::Kernel transpose(radix_kernel, radix_kernel_length, "transpose", param_string);
        transpose.compile();
        ocl::Kernel prefix(radix_kernel, radix_kernel_length, "prefix_sum", param_string);
        prefix.compile();
        ocl::Kernel reduce(radix_kernel, radix_kernel_length, "reduce", param_string);
        reduce.compile();
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix", param_string);
        radix.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            gpu::gpu_mem_32u bs_gpu;
            bs_gpu.resizeN(n);

            gpu::gpu_mem_32u counters_gpu, counters_t1_gpu, counters_t2_gpu, prefix_gpu;
            counters_gpu.resizeN(CNT_TOTAL);
            prefix_gpu.resizeN(CNT_TOTAL);
            counters_t1_gpu.resizeN(CNT_TOTAL);
            counters_t2_gpu.resizeN(CNT_TOTAL);
            std::vector<unsigned int> zeros(CNT_TOTAL, 0);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            for (unsigned int curr_digit = 1; curr_digit <= N_DIGITS; curr_digit++)
            {
                counters_gpu.writeN(zeros.data(), CNT_TOTAL);
                prefix_gpu.writeN(zeros.data(), CNT_TOTAL);
                counters_t1_gpu.writeN(zeros.data(), CNT_TOTAL);
                counters_t2_gpu.writeN(zeros.data(), CNT_TOTAL);

                //0. Small sort inside blocks
                //std::cout << "Small sort" << std::endl;
                small_sort.exec(gpu::WorkSize(WG_SIZE, n), as_gpu, n, curr_digit);

                //std::vector<unsigned int> temp1(n);
                //as_gpu.readN(temp1.data(), temp1.size());
                //for (int i = 0; i < temp1.size(); i++)
                //{
                //    std::cout << temp1[i] << ' ';
                //}
                //std::cout << std::endl;

                //1. Calculate counters;
                //std::cout << "Counters" << std::endl;
                counters.exec(gpu::WorkSize(WG_SIZE, n), as_gpu, n, counters_gpu, curr_digit);

                //std::vector<unsigned int> temp2(N_WG * (1 << BITS_IN_DIGIT));
                //counters_gpu.readN(temp2.data(), temp2.size());
                //for (int i = 0; i < temp2.size(); i++)
                //{
                //    if (i % (1 << BITS_IN_DIGIT) == 0)
                //    {
                //        std::cout << std::endl;
                //    }
                //    std::cout << temp2[i] << ' ';
                //}
                //std::cout << std::endl;

                //2. Transpose counters
                gpu::WorkSize ws(16, 16, CNT_SIZE, N_WG);
                //std::cout << "Transpose" << std::endl;
                transpose.exec(ws, counters_gpu, counters_t1_gpu, N_WG, CNT_SIZE);

                //std::vector<unsigned int> temp3(N_WG * (1 << BITS_IN_DIGIT));
                //counters_t1_gpu.readN(temp3.data(), temp3.size());
                //for (int i = 0; i < temp3.size(); i++)
                //{
                //    if (i % N_WG == 0)
                //    {
                //        std::cout << std::endl;
                //    }
                //    std::cout << temp3[i] << ' ';
                //}
                //std::cout << std::endl;

                //3. Calculate prefix sums
                //std::cout << "Prefix" << std::endl;
                for (unsigned int take_id = 1; take_id <= CNT_TOTAL; take_id <<= 1)
                {
                    prefix.exec(gpu::WorkSize(WG_SIZE, CNT_TOTAL), counters_t1_gpu, prefix_gpu, CNT_TOTAL, take_id);
                    reduce.exec(gpu::WorkSize(WG_SIZE, gpu::divup(CNT_TOTAL, take_id)), counters_t1_gpu, counters_t2_gpu, CNT_TOTAL / take_id);
                    std::swap(counters_t1_gpu, counters_t2_gpu);
                }
                //prefix.exec(gpu::WorkSize(WG_SIZE, CNT_SIZE), prefix1_gpu, CNT_SIZE, N_WG);

                //std::vector<unsigned int> temp4(N_WG * (1 << BITS_IN_DIGIT));
                //prefix_gpu.readN(temp4.data(), temp4.size());
                //for (int i = 0; i < temp4.size(); i++)
                //{
                //    if (i % N_WG == 0)
                //    {
                //        std::cout << std::endl;
                //    }
                //    std::cout << temp4[i] << ' ';
                //}
                //std::cout << std::endl;

                //4. Assign elements to positions
                //std::cout << "Radix" << std::endl;
                radix.exec(gpu::WorkSize(WG_SIZE, n), as_gpu, n, counters_gpu, prefix_gpu, N_WG, CNT_SIZE, curr_digit, bs_gpu);
                std::swap(as_gpu, bs_gpu);

                //std::vector<unsigned int> temp5(n);
                //as_gpu.readN(temp5.data(), temp5.size());
                //for (int i = 0; i < temp5.size(); i++)
                //{
                //    std::cout << temp5[i] << ' ';
                //}
                //std::cout << std::endl;
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000. / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    std::cout << "Ok!" << std::endl;
    return 0;
}
