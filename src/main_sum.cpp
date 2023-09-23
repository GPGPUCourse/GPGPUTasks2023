#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

using namespace std;

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

static int benchmarkingIters = 10;
static unsigned int n = 100*1000*1000;
static unsigned int reference_sum = 0;

void measure(string name, gpu::gpu_mem_32u as, gpu::WorkSize size);

int main(int argc, char **argv)
{
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
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
    std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
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
    std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
    // TODO: implement on OpenCL
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();
    gpu::gpu_mem_32u buffer;
    buffer.resizeN(n);
    buffer.writeN(as.data(), n);

    const unsigned groupSize = 128;
    const unsigned globalSize = (n + groupSize - 1) / groupSize * groupSize;
    const unsigned valuesPerItem = 64;
    measure("simple", buffer, gpu::WorkSize(groupSize, n));
    measure("cycle", buffer, gpu::WorkSize(groupSize, globalSize / valuesPerItem));
    measure("cycle_break", buffer, gpu::WorkSize(groupSize, globalSize / valuesPerItem));
    measure("smart_cycle", buffer, gpu::WorkSize(groupSize, globalSize / valuesPerItem));
    measure("local_mem", buffer, gpu::WorkSize(groupSize, globalSize));
    measure("tree", buffer, gpu::WorkSize(groupSize, globalSize));
    }
    /* Perfomance notes:
     Ну, у меня интеловский gpu и размер варп сайза мне пишет 8
     Поэтому в принципе не удивительно что cpu с omp оказался быстрее чем gpu.
     В рамках самого gpu coalesed доступ действительно оказался быстрее чем обычный
     Ну супер оптимизации подсчета по дереву не сработали скорее всего потому что gpu слабый
     */
}

void measure(string name, gpu::gpu_mem_32u as, gpu::WorkSize size) {
    ocl::Kernel kernel(sum_kernel, sum_kernel_length, name);
    kernel.compile();
    gpu::gpu_mem_32u sum;
    sum.resizeN(1);
    unsigned zero = 0;
    timer t;
    for (int i = 0; i < benchmarkingIters; ++i) {
        sum.writeN(&zero, 1);
        kernel.exec(size, as, sum, n);
        unsigned result = 0;
        sum.readN(&result, 1);
        EXPECT_THE_SAME(reference_sum, result, "GPU result should be consistent!");
        t.nextLap();
    }
    std::cout << "GPU " << name << ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU " << name << ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
}
