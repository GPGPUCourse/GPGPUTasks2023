#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"

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

    uint32_t reference_sum = 0;
    uint32_t n = 100 * 1000 * 1000;
    std::vector<uint32_t> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (uint32_t) r.next(0, std::numeric_limits<uint32_t>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            uint32_t sum = 0;
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
            uint32_t sum = 0;
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

    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);

        {
            ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_gpu_atomic");

            bool printLog = true;
            kernel.compile(printLog);
            {
                gpu::WorkSize workSize(128, n);

                timer t;
                uint32_t sum = 0;
                gpu::gpu_mem_32u gpu_sum;
                gpu_sum.resizeN(1);

                for (int i = 0; i < benchmarkingIters; ++i) {
                    uint32_t zero = 0;
                    gpu_sum.writeN(&zero, 1);

                    kernel.exec(workSize,
                                as_gpu, // __global const uint32_t *as_gpu
                                gpu_sum // __global uint32_t *sum
                    );

                    gpu_sum.readN(&sum, 1);
                    EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");

                    t.nextLap();
                }
                std::cout << "GPU atomic:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU atomic:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
            }
        }
    }
}
