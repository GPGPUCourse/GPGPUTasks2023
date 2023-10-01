#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/matrix_multiplication_cl.h"

#include <vector>
#include <iostream>
#include <stdexcept>


int benchmarkingIters = 20; // TODO пока тестируетесь удобно выставить единицу
unsigned int n = 1024;
unsigned int m = 512;
unsigned int l = 256;
const double gflops = ((size_t) n * m * l * 2) / (1000.0 * 1000 * 1000); // умножить на два, т.к. операция сложения и умножения

std::vector<float> as(n*m, 0);
std::vector<float> bs(m*l, 0);
std::vector<float> cs(n*l, 0);

gpu::gpu_mem_32f as_gpu, bs_gpu, cs_gpu;

void test(ocl::Kernel kernel, std::string name, std::vector<float> cs_cpu_reference, int w = 1) {
    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        // TODO
        kernel.exec(gpu::WorkSize(16, 16, l / w, n), as_gpu, bs_gpu, cs_gpu, n, m, l);
        t.nextLap();
    }
    std::cout << name << " GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << name << " GPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;

    cs_gpu.readN(cs.data(), n*l);

    // Проверяем корректность результатов
    double diff_sum = 0;
    for (int i = 0; i < n * l; ++i) {
        double a = cs[i];
        double b = cs_cpu_reference[i];
        if (a != 0.0 || b != 0.0) {
            double diff = fabs(a - b) / std::max(fabs(a), fabs(b));
            diff_sum += diff;
        }
    }

    double diff_avg = diff_sum / (n * l);
    std::cout << name << " Average difference: " << diff_avg * 100.0 << "%" << std::endl;
    if (diff_avg > 0.01) {
        std::cerr << "Too big difference!" << std::endl;
        exit(1);
    }
}

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    FastRandom r(n+m+l);
    for (unsigned int i = 0; i < as.size(); ++i) {
        as[i] = r.nextf();
    }
    for (unsigned int i = 0; i < bs.size(); ++i) {
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << ", m=" << m << ", k=" << l << std::endl;

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            for (int j = 0; j < n; ++j) {
                for (int i = 0; i < l; ++i) {
                    float sum = 0.0f;
                    for (int k = 0; k < m; ++k) {
                        sum += as.data()[j * m + k] * bs.data()[k * l + i];
                    }
                    cs.data()[j * l + i] = sum;
                }
            }
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << gflops / t.lapAvg() << " GFlops" << std::endl;
    }

    const std::vector<float> cs_cpu_reference = cs;

    as_gpu.resizeN(n*m);
    bs_gpu.resizeN(m*l);
    cs_gpu.resizeN(n*l);

    as_gpu.writeN(as.data(), n*m);
    bs_gpu.writeN(bs.data(), m*l);

    ocl::Kernel naiveKernel(matrix_multiplication, matrix_multiplication_length, "simple");
    naiveKernel.compile();

    ocl::Kernel localmemKernel(matrix_multiplication, matrix_multiplication_length, "localmem");
    localmemKernel.compile();

    ocl::Kernel moreworkKernel(matrix_multiplication, matrix_multiplication_length, "morework");
    moreworkKernel.compile();

    test(naiveKernel, "naive", cs_cpu_reference);
    test(localmemKernel, "localmem", cs_cpu_reference);
    test(moreworkKernel, "morework", cs_cpu_reference, 4);

    return 0;
}
