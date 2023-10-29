#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"
#include "cl/matrix_transpose_cl.h"
#include "cl/prefix_sum_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>
#include <cassert>


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

    static const int MaxBits = 32;

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
//    int benchmarkingIters = 1;
//    unsigned int n = 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
//        as[i] = (unsigned) r.next(0, (1 << MaxBits) - 1);
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
        std::cout << "CPU: " << (n / 1000.0 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }
    gpu::gpu_mem_32u as_gpu;
    as_gpu.resizeN(n);
    gpu::gpu_mem_32u bs_gpu;
    bs_gpu.resizeN(n);

    auto showArr = [&](const std::string &title, const gpu::gpu_mem_32u &arr) {
        std::vector<unsigned> vec(arr.number());
        arr.readN(vec.data(), arr.number());
        std::cout << title << ":\n";
        std::cout << "[";
        for (size_t i = 0; i < arr.number(); i++) {
            std::cout << vec[i];
            std::cout << (i + 1 == arr.number() ? ']' : ' ');
        }
        std::cout << '\n';
    };

    auto show2dArr = [&](const std::string &title, const gpu::gpu_mem_32u &arr, unsigned cols, unsigned rows) {
        assert(cols * rows == arr.number());
        std::vector<unsigned> vec(arr.number());
        arr.readN(vec.data(), arr.number());
        std::cout << title << ":\n";
        std::cout << "[\n";
        for (size_t i = 0; i < rows; i++) {
            for (size_t j = 0; j < cols; j++) {
                std::cout << '\t' << vec[i * cols + j] << (j + 1 == cols ? ']' : ' ');
            }
            std::cout << '\n';
        }
        std::cout << "]\n";
    };

    {
//        static const int WorkGroupSize = 128;
        static const int WorkGroupSize = 128;
        static const int BlockBits = 3;
        static const int BlockNumbers = (1 << BlockBits);
        static const int MatrixTransposeWorkGroupSize = BlockNumbers;

        std::string defines = "-D WORK_GROUP_SIZE=" + to_string(WorkGroupSize) +
                              " -D BLOCK_BITS=" + to_string(BlockBits) +
                              " -D MT_WORK_GROUP_SIZE=" + to_string(MatrixTransposeWorkGroupSize);
        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix", defines);
        radix.compile();

        ocl::Kernel matrix_transpose(matrix_transpose_kernel, matrix_transpose_kernel_length, "matrix_transpose",
                                     defines);
        matrix_transpose.compile();

        ocl::Kernel sort_for_each_work_group(radix_kernel, radix_kernel_length,
                                             "sort_for_each_work_group", defines);
        sort_for_each_work_group.compile();

        ocl::Kernel calculate_counts_in_each_block(radix_kernel, radix_kernel_length, "calculate_counts_in_each_block",
                                                   defines);
        calculate_counts_in_each_block.compile();

        ocl::Kernel prefix_sum(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum");

        assert(n % WorkGroupSize == 0);
        unsigned countsBlocks = n / WorkGroupSize;
        unsigned countLength = BlockNumbers * countsBlocks;
        gpu::gpu_mem_32u counts;
        counts.resizeN(countLength);
        gpu::gpu_mem_32u countsPrefSum;
        countsPrefSum.resizeN(countLength);
        gpu::gpu_mem_32u countsHelper;
        countsHelper.resizeN(countLength);

        auto *bsGpuPtr = &bs_gpu;
        auto *asGpuPtr = &as_gpu;

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            asGpuPtr = &as_gpu;
            bsGpuPtr = &bs_gpu;

//            showArr("Start sorting arr", as_gpu);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

            for (int shift = 0; shift < MaxBits; shift += BlockBits) {

//                std::cout << "Start for shift=" << shift << '\n';

                // get counts
                calculate_counts_in_each_block.exec(gpu::WorkSize(WorkGroupSize, n), *asGpuPtr, n, counts, shift);
//                show2dArr("counts", counts, BlockNumbers, countsBlocks);

                // transpose counts
                matrix_transpose.exec(
                        gpu::WorkSize(MatrixTransposeWorkGroupSize, MatrixTransposeWorkGroupSize, BlockNumbers,
                                      countsBlocks),
                        counts, countsPrefSum, BlockNumbers, countsBlocks);

//                show2dArr("counts transposed", countsPrefSum, countsBlocks, BlockNumbers);
//                showArr("Counts transposed 1d", countsPrefSum);

                // prefix sum for counts
                auto *countsPrefSumPtr = &countsPrefSum;
                auto *countsHelperPtr = &countsHelper;
                for (unsigned blockSize = 1; blockSize <= countLength; blockSize *= 2) {
                    prefix_sum.exec(gpu::WorkSize(WorkGroupSize, countLength), *countsPrefSumPtr, *countsHelperPtr,
                                    countLength, blockSize);
                    std::swap(countsPrefSumPtr, countsHelperPtr);
                }

//                show2dArr("countsPrefSum", *countsPrefSumPtr, countsBlocks, BlockNumbers);

//                showArr("as_gpu", *asGpuPtr);
                // sort each work group
                sort_for_each_work_group.exec(gpu::WorkSize(WorkGroupSize, n), *asGpuPtr, n, shift);

//                showArr("as after blocks' bitonic sort", *asGpuPtr);

//                showArr("counts before finish", counts);
//                showArr("countsPrefSum before finish", *countsPrefSumPtr);
                // finish
                radix.exec(gpu::WorkSize(WorkGroupSize, n), *asGpuPtr, n, counts, *countsPrefSumPtr, *bsGpuPtr, shift);

//                showArr("result arr", *bsGpuPtr);
                std::swap(asGpuPtr, bsGpuPtr);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

        asGpuPtr->readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
//        std::cout << i << ' ' << cpu_sorted[i] << ' ' << as[i] << '\n';
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
