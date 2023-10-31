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
    transposeKernel.exec(WorkSize(4u, 4u, m, n), output, input, n, m);
}

void prefix_sum(buffer output, buffer input, unsigned n) {
    static auto pickKernel = createPickKernel();
    static auto reduceKernel = createReduceKernel();

    pickKernel.exec(WorkSize(groupSize, n), output, input, 0u);
    for (unsigned k = 1u; 1u << k <= n; ++k) {
        reduceKernel.exec(WorkSize(std::max(1u, groupSize >> k), n >> k), input, k);
        pickKernel.exec(WorkSize(groupSize, n), output, input, k);
    }
}

void merge_sort(buffer &output, buffer input, unsigned n, unsigned shift) {
    static auto mergeKernel = createMergeSortKernel();
    static auto copy = buffer::createN(n);
    input.copyToN(copy, n);

    for (unsigned size = 1; size < groupSize; size *= 2u) {
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

    const unsigned tableRows = n / groupSize;
    const unsigned tableCols = 1 << nbits;
    const unsigned tableSize = tableRows * tableCols;

    std::vector<unsigned> zeros(n, 0);

    {
        auto input = buffer::createN(n);
        auto counters = buffer::createN(tableSize);
        auto counters_t = buffer::createN(tableSize);
        auto aggregated_t = buffer::createN(tableSize);
        auto aggregated = buffer::createN(tableSize);
        auto sorted = buffer::createN(n);
        auto output = buffer::createN(n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            input.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            for (unsigned shift = 0; shift < 32; shift += nbits) {
                counters.writeN(zeros.data(), tableSize);
                aggregated_t.writeN(zeros.data(), tableSize);

                count(counters, input, n, shift);
                transpose(counters_t, counters, tableRows, tableCols);
                prefix_sum(aggregated_t, counters_t, tableSize);
                transpose(aggregated, aggregated_t, tableCols, tableRows);

                merge_sort(sorted, input, n, shift);
                radix(output, sorted, n, aggregated, tableRows, shift);

                output.swap(input);
            }

            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        output.swap(input);
        output.readN(as.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }
    return 0;
}
