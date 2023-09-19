#include "cl/sum_cl.h"

#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

#include "libgpu/context.h"
#include "libgpu/device.h"
#include "libgpu/shared_device_buffer.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000;//*1000;
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
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+ : sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    // Нам пригодятся ещё 2 дополнительных буфера, помимо входного
    gpu::gpu_mem_32u buffers[3];
    for (auto &buf : buffers) {
        buf.resizeN(n);
    }
    // Не можем переиспользовать USE_HOST в этой библиотеке
    buffers[0].writeN(&as[0], n);

    const size_t WORKGROUP_SIZE = 128;
    const size_t VALUES_PER_WORKITEM = 64;

    std::string defines;
    {
        std::ostringstream oss;
        oss << "-DWORKGROUP_SIZE=" << WORKGROUP_SIZE << " -DVALUES_PER_WORKITEM=" << VALUES_PER_WORKITEM;
        defines = oss.str();
    }

    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum1_atomic", defines);
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cl_uint sum = 0;
            buffers[1].writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, n), buffers[0].clmem(), buffers[1].clmem(), cl_ulong(n));
            buffers[1].readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        double lapAvg = t.lapAvg();
        double lapStd = t.lapStd();

        double speed = n / (1e6 * lapAvg);

        std::cout << "Atomic sum: " << lapAvg << "+-" << lapStd << " s\n";
        std::cout << "Atomic sum: " << speed << " M/s\n";
    }

    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum2_loop", defines);
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cl_uint sum = 0;
            buffers[1].writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, n), buffers[0].clmem(), buffers[1].clmem(), cl_ulong(n));
            buffers[1].readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        double lapAvg = t.lapAvg();
        double lapStd = t.lapStd();

        double speed = n / (1e6 * lapAvg);

        std::cout << "Loop sum: " << lapAvg << "+-" << lapStd << " s\n";
        std::cout << "Loop sum: " << speed << " M/s\n";
    }
}
