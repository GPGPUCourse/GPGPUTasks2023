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

#define Nbits 2u
#define Ncols (1u << Nbits)
#define Mask ((1u << Nbits) - 1)
#define GetKey(value, shift) (((value) >> (shift)) & Mask)


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

const unsigned groupSize = 256;
const unsigned nbits = 2;

typedef gpu::gpu_mem_32u buffer;

using ocl::Kernel;
using gpu::WorkSize;


static Kernel createCountKernel() {
    Kernel count(radix_kernel, radix_kernel_length, "count");
    count.compile();
    return count;
}
static Kernel createTransposeKernel() {
    Kernel transpose(radix_kernel, radix_kernel_length, "transpose");
    transpose.compile();
    return transpose;
}
static Kernel createPickKernel() {
    Kernel pick(radix_kernel, radix_kernel_length, "pick");
    pick.compile();
    return pick;
}
static Kernel createReduceKernel() {
    Kernel reduce(radix_kernel, radix_kernel_length, "reduce");
    reduce.compile();
    return reduce;
}
static Kernel createMergeSortKernel() {
    Kernel merge(radix_kernel, radix_kernel_length, "merge");
    merge.compile();
    return merge;
}
static Kernel createRadixSortKernel() {
    Kernel radix(radix_kernel, radix_kernel_length, "radix");
    radix.compile();
    return radix;
}

void count(buffer output, buffer input, unsigned n, unsigned shift) {
    static auto countKernel = createCountKernel();
    countKernel.exec(WorkSize(groupSize, n), output, input, shift);
}

void transpose(buffer output, buffer input, unsigned n, unsigned m) {
    static auto transposeKernel = createTransposeKernel();
    transposeKernel.exec(WorkSize(4, 4, m, n), output, input, n, m);
}

void prefix_sum(buffer output, buffer input, unsigned n) {
    static auto pickKernel = createPickKernel();
    static auto reduceKernel = createReduceKernel();

    pickKernel.exec(WorkSize(groupSize, n), output, input, 0);
    for (int k = 1; 1 << k <= n; ++k) {
        reduceKernel.exec(WorkSize(std::max(1u, groupSize >> k), n >> k), input, k);
        pickKernel.exec(WorkSize(groupSize, n), output, input, k);
    }
}

void merge_sort(buffer &output, buffer input, unsigned n, unsigned shift) {
    static auto mergeKernel = createMergeSortKernel();
    static auto copy = buffer::createN(n);
    input.copyToN(copy, n);

    for (unsigned size = 1; size < groupSize; size *= 2) {
        mergeKernel.exec(WorkSize(groupSize, n), output, copy, size, shift);
        output.swap(copy);
    }
    // no need to check for log(n) this time ;)
    output.swap(copy);
}

void radix(buffer output, buffer input, unsigned n, buffer offsets, unsigned rows, unsigned shift) {
    static auto radixKernel = createRadixSortKernel();
    radixKernel.exec(WorkSize(groupSize, n), output, input, n, offsets, rows, shift);
}


int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 1;
    unsigned int n = 1024;
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


    const unsigned tableRows = n / groupSize;
    const unsigned tableCols = 1 << nbits;
    const unsigned tableSize = tableRows * tableCols;

    std::vector<unsigned> result(tableSize);
    std::vector<unsigned> control(tableSize, 0);

    std::vector<unsigned> zeros(n, 0);

//    Kernel radix(radix_kernel, radix_kernel_length, "radix");
//    radix.compile();
//    Kernel count(radix_kernel, radix_kernel_length, "count");
//    count.compile();
//    Kernel pick(radix_kernel, radix_kernel_length, "pick");
//    pick.compile();
//    Kernel reduce(radix_kernel, radix_kernel_length, "reduce");
//    reduce.compile();

    auto input = buffer::createN(n);
    auto counters = buffer::createN(tableSize);
    auto counters_t = buffer::createN(tableSize);

    input.writeN(as.data(), n);
    counters.writeN(zeros.data(), tableSize);

    count(counters, input, n, 0);

//    count.exec(WorkSize(groupSize, n), counters, input, 0);
    counters.readN(result.data(), tableSize);

    for (int i = 0; i < n; ++i) {
        control[(as[i] & 0x3) + 4 * (i / groupSize)]++;
    }
    for (int i = 0; i < tableSize; ++i) {
        EXPECT_THE_SAME(control[i], result[i], "GPU results should be equal to CPU results!");
    }

    transpose(counters_t, counters, tableRows, tableCols);

    counters_t.readN(result.data(), tableSize);

    std::vector<unsigned> control_t(tableSize);
    for (int i = 0; i < tableRows; ++i) {
        for (int j = 0; j < tableCols; ++j) {
            control_t[j * tableRows + i] = control[i * tableCols + j];
        }
    }
    for (int i = 0; i < tableCols; ++i) {
        for (int j = 0; j < tableRows; ++j) {
            EXPECT_THE_SAME(control_t[i * tableRows + j], result[i * tableRows + j], "GPU results should be equal to CPU results!");
        }
    }
   control = control_t;

    auto aggregated_t = buffer::createN(tableSize);
    aggregated_t.writeN(zeros.data(), tableSize);

//    aggregated.readN(result.data(), counterCount);
//    pick.exec(WorkSize(groupSize, counterCount), aggregated, counters, 0);
//    aggregated.readN(result.data(), counterCount);
//    for (int k = 1; 1u << k <= counterCount; ++k) {
//        reduce.exec(WorkSize(std::max(1u, groupSize >> k), counterCount >> k), counters, k);
//        pick.exec(WorkSize(groupSize, counterCount), aggregated, counters, k);
//        aggregated.readN(result.data(), counterCount);
//    }
    prefix_sum(aggregated_t, counters_t, tableSize);

    aggregated_t.readN(result.data(), tableSize);

    for (int i = 0; i < tableSize; ++i) {
        if (i) control[i] += control[i - 1];
    }
    for (int i = 0; i < tableSize; ++i) {
        EXPECT_THE_SAME(control[i], result[i], "GPU results should be equal to CPU results!");
    }

    auto sorted = buffer::createN(n);
    merge_sort(sorted, input, n, 0);

    std::vector<unsigned> control2(n, 0);
    sorted.readN(control2.data(), n);

    for (int i = 0; i < n; ++i) {
        if (std::find(control2.begin(), control2.end(), as[i]) == control2.end()) {
            throw std::exception();
        }
//        for (int j = 0; j < n; ++j) {
//            if (input[i] == control2[j]) bre
//        }
    }

    auto aggregated = buffer::createN(tableSize);
    transpose(aggregated, aggregated_t, tableCols, tableRows);

    std::vector<unsigned> a(n, 0);
    sorted.readN(a.data(), n);
    std::vector<unsigned> offsets(tableSize, 0);
    aggregated.readN(offsets.data(), tableSize);

    for (int i = 0; i < tableCols; ++i) {
        for (int j = 0; j < tableRows; ++j) {
            control_t[j * tableCols + i] = control[i * tableRows + j];
        }
    }
    for (int i = 0; i < tableRows; ++i) {
        for (int j = 0; j < tableCols; ++j) {
            EXPECT_THE_SAME(control_t[i * tableCols + j], offsets[i * tableCols + j], "GPU results should be equal to CPU results!");
        }
    }

    const unsigned rows = tableRows;

    auto output1 = buffer::createN(n);
    output1.writeN(zeros.data(), n);
    std::vector<unsigned> output(n, 0);
    radix(output1, sorted, n, aggregated, tableRows, 0);
//    radix.exec(WorkSize(groupSize, n), output1, input, n, aggregated_t, rows, shift);
    output1.readN(output.data(), n);


    unsigned shift = 2;
    counters.writeN(zeros.data(), tableSize);
    aggregated_t.writeN(zeros.data(), tableSize);
    output1.swap(input);

    count(counters, input, n, shift);
    transpose(counters_t, counters, tableRows, tableCols);
    prefix_sum(aggregated_t, counters_t, tableSize);
    merge_sort(sorted, input, n, shift);
    transpose(aggregated, aggregated_t, tableCols, tableRows);
    radix(output1, sorted, n, aggregated, tableRows, shift);

    shift = 4;
    counters.writeN(zeros.data(), tableSize);
    aggregated_t.writeN(zeros.data(), tableSize);
    output1.swap(input);

    count(counters, input, n, shift);
    transpose(counters_t, counters, tableRows, tableCols);
    prefix_sum(aggregated_t, counters_t, tableSize);
    merge_sort(sorted, input, n, shift);
    transpose(aggregated, aggregated_t, tableCols, tableRows);
    radix(output1, sorted, n, aggregated, tableRows, shift);

    output1.readN(output.data(), n);


    auto control3 = as;

//    for (int id = 0; id < n; ++id) {
//        unsigned lid = id % groupSize;
//        unsigned group = id / groupSize;
//
//        unsigned num = GetKey(a[id], shift);
//
//        unsigned offsetLower = 0, offsetEqual = 0;
//        if (group == 0) {
//            if (num) offsetLower = offsets[(rows - 1) * Ncols + num - 1];
//        } else {
//            offsetLower = offsets[(group - 1) * Ncols + num];
//        }
//        for (int i = 0; i < num; ++i) {
//            offsetEqual += offsets[group * Ncols + i];
//            if (group == 0) {
//                if (i) offsetEqual -= offsets[(rows - 1) * Ncols + i - 1];
//            } else {
//                offsetEqual -= offsets[(group - 1) * Ncols + i];
//            }
//        }
//        if (output[lid - offsetEqual + offsetLower]) {
//            printf("Already been there: %d\n", lid - offsetEqual + offsetLower);
//        }
//        output[lid - offsetEqual + offsetLower] = a[id];
//    }

    std::stable_sort(control3.begin(), control3.end(), [](unsigned x, unsigned y) {
        return (x & 0x3) < (y & 0x3);
    });
    std::stable_sort(control3.begin(), control3.end(), [](unsigned x, unsigned y) {
        return (x >> 2 & 0x3) < (y >> 2 & 0x3);
    });
    std::stable_sort(control3.begin(), control3.end(), [](unsigned x, unsigned y) {
        return (x >> 4 & 0x3) < (y >> 4 & 0x3);
    });

//    output.readN(result2.data(), n);

    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(output[i], control3[i], "GPU results should be equal to CPU results!");
    }


//    gpu::gpu_mem_32u as_gpu, pre_sorted;
//    as_gpu.resizeN(n);
//    pre_sorted.resizeN(n);
//
//    gpu::gpu_mem_32u counts, counts_t;
//    counts.resizeN(N);
//    counts_t.resizeN(N);
//    std::vector<unsigned> zeros(N, 0);
//
//    gpu::gpu_mem_32u result_gpu, result_gpu_t;
//    result_gpu.resizeN(N);
//    result_gpu_t.resizeN(N);
//
//    gpu::gpu_mem_32u f;
//    f.resizeN(n);
//
//    {
//        ocl::Kernel merge(radix_kernel, radix_kernel_length, "merge");
//        merge.compile();
//
//        ocl::Kernel getCounts(radix_kernel, radix_kernel_length, "getCounts");
//        getCounts.compile();
//
//        ocl::Kernel transpose(radix_kernel, radix_kernel_length, "transpose");
//        transpose.compile();
//        ocl::Kernel radix(radix_kernel, radix_kernel_length, "radix");
//        radix.compile();
//
//        ocl::Kernel prefix(radix_kernel, radix_kernel_length, "prefix");
//        ocl::Kernel reduce(radix_kernel, radix_kernel_length, "reduce");
//        prefix.compile();
//        reduce.compile();
//
//        timer t;
//        for (int iter = 0; iter < benchmarkingIters; ++iter) {
//            as_gpu.writeN(as.data(), n);
//
//            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
//            for (unsigned mask = 0x3, shift = 0; shift < 32; shift += 2, mask <<= 2) {
//                counts.writeN(zeros.data(), N);
//                // count
//                getCounts.exec(gpu::WorkSize(groupSize, n), as_gpu, counts, mask, shift);
//                // transpose
//                transpose.exec(gpu::WorkSize(16, 16, ncounters, groupSize), counts, counts_t, groupSize, ncounters);
//                // prefix sum
//                for (unsigned k = 0; (1u << k) <= N; ++k) {
//                    reduce.exec(gpu::WorkSize(groupSize, N), counts_t, k);
//                    prefix.exec(gpu::WorkSize(groupSize, N), counts_t, result_gpu, k);
//                }
//                // transpose again
//                transpose.exec(gpu::WorkSize(16, 16, groupSize, ncounters), result_gpu, result_gpu_t, ncounters, groupSize);
//                // merge sort
//                for (unsigned k = n, i = 1; i < groupSize; k /= 2, i *= 2) {
//                    merge.exec(gpu::WorkSize(groupSize, n), pre_sorted, as_gpu, n, k, mask, shift);
//                    pre_sorted.swap(as_gpu);
//                }
//                // radix sort
//                radix.exec(gpu::WorkSize(groupSize, n), as_gpu, n, counts, result_gpu_t, N, f, mask, shift);
//
////                std::vector<unsigned> res(N);
////                std::vector<unsigned> res_t(N);
////                counts.readN(res.data(), N);
////                counts_t.readN(res_t.data(), N);
////                f.swap(as_gpu);
//            }
//            std::vector<unsigned> res(n);
//            f.readN(res.data(), n);
//
//            t.nextLap();
////            std::cout << res << std::endl;
//        }
//        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
//        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
//
//        f.readN(as.data(), n);
//    }

    // Проверяем корректность результатов
//    for (int i = 0; i < n; ++i) {
//        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
//    }
    return 0;
}
