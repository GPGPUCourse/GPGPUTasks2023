#include "cl/sum_cl.h"

#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "libgpu/context.h"
#include "libgpu/device.h"
#include "libgpu/shared_device_buffer.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv) {
    int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
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
    gpu::gpu_mem_32u bufA;
    gpu::gpu_mem_32u buf1;
    bufA.resizeN(n);
    buf1.resizeN(1);
    // Не можем переиспользовать USE_HOST в этой библиотеке
    bufA.writeN(&as[0], n);

    const size_t WORKGROUP_SIZE = 64;
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
            buf1.writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, n), bufA.clmem(), buf1.clmem(), cl_ulong(n));
            buf1.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        t.stop();
        double lapAvg = t.lapAvg();
        double lapStd = t.lapStd();

        double speed = n / (1e6 * lapAvg);

        std::cout << "Atomic: " << lapAvg << "+-" << lapStd << " s\n"
                  << "Atomic: " << speed << " M/s\n";
    }

    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum2_loop", defines);
        size_t workSize = (n + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM;
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cl_uint sum = 0;
            buf1.writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, workSize), bufA.clmem(), buf1.clmem(), cl_ulong(n));
            buf1.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        t.stop();
        double lapAvg = t.lapAvg();
        double lapStd = t.lapStd();

        double speed = n / (1e6 * lapAvg);

        std::cout << "Loop: " << lapAvg << "+-" << lapStd << " s\n"
                  << "Loop: " << speed << " M/s\n";
    }

    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum3_loop_coalesced", defines);
        size_t workSize = (n + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM;
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cl_uint sum = 0;
            buf1.writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, workSize), bufA.clmem(), buf1.clmem(), cl_ulong(n));
            buf1.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        t.stop();
        double lapAvg = t.lapAvg();
        double lapStd = t.lapStd();

        double speed = n / (1e6 * lapAvg);

        std::cout << "Coalesced loop: " << lapAvg << "+-" << lapStd << " s\n"
                  << "Coalesced loop: " << speed << " M/s\n";
    }

    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum3_loop_coalesced", defines);
        size_t workSize = (n + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM;
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cl_uint sum = 0;
            buf1.writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, workSize), bufA.clmem(), buf1.clmem(), cl_ulong(n));
            buf1.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        t.stop();
        double lapAvg = t.lapAvg();
        double lapStd = t.lapStd();

        double speed = n / (1e6 * lapAvg);

        std::cout << "Coalesced loop: " << lapAvg << "+-" << lapStd << " s\n"
                  << "Coalesced loop: " << speed << " M/s\n";
    }

    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum4_local_mem", defines);
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cl_uint sum = 0;
            buf1.writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, n), bufA.clmem(), buf1.clmem(), cl_ulong(n));
            buf1.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        t.stop();
        double lapAvg = t.lapAvg();
        double lapStd = t.lapStd();

        double speed = n / (1e6 * lapAvg);

        std::cout << "Local memory loop: " << lapAvg << "+-" << lapStd << " s\n"
                  << "Local memory loop: " << speed << " M/s\n";
    }

    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum5_tree_local", defines);
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cl_uint sum = 0;
            buf1.writeN(&sum, 1);
            kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, n), bufA.clmem(), buf1.clmem(), cl_ulong(n));
            buf1.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        t.stop();
        double lapAvg = t.lapAvg();
        double lapStd = t.lapStd();

        double speed = n / (1e6 * lapAvg);

        std::cout << "Local memory tree: " << lapAvg << "+-" << lapStd << " s\n"
                  << "Local memory tree: " << speed << " M/s\n";
    }

    {
        ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum6_tree_global", defines);

        gpu::gpu_mem_32u bufSrc;
        gpu::gpu_mem_32u bufDst;
        bufSrc.resizeN(n);
        bufDst.resizeN(n);

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            bufA.copyToN(bufSrc, n);
            for (size_t workSize = n; workSize > 1; workSize = (workSize + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE) {
                kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, workSize),
                            bufSrc.clmem(), bufDst.clmem(), cl_ulong(workSize));
                std::swap(bufSrc, bufDst);
            }
            cl_uint sum = 0;
            bufSrc.readN(&sum, 1);
            EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
            t.nextLap();
        }
        t.stop();
        double lapAvg = t.lapAvg();
        double lapStd = t.lapStd();

        double speed = n / (1e6 * lapAvg);

        std::cout << "Global memory tree: " << lapAvg << "+-" << lapStd << " s\n"
                  << "Global memory tree: " << speed << " M/s\n";
    }
}
