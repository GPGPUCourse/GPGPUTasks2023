#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <sstream>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"


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

    constexpr int benchmarkingIters = 1;// 10;
    constexpr cl_uint max_n = 1 << 24;

    constexpr cl_uint workGroupSize = 64;
    std::ostringstream defines;
    defines << "-DWORK_GROUP_SIZE=" << workGroupSize;

    for (cl_uint n = 4096; n <= max_n; n *= 4) {
        std::cout << "______________________________________________" << std::endl;
        cl_uint values_range = std::min<cl_uint>(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n=" << n << " values in range: [" << 0 << "; " << values_range << "]" << std::endl;

        std::vector<cl_uint> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = r.next(0, values_range);
        }

        std::vector<cl_uint> result(n, 0);
        for (int i = 0; i < n; ++i) {
            result[i] = as[i];
            if (i) {
                result[i] += result[i - 1];
            }
        }
        const std::vector<cl_uint> reference_result = result;

        {
            for (int i = 0; i < n; ++i) {
                result[i] = as[i];
                if (i) {
                    result[i] += result[i - 1];
                }
            }
            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(reference_result[i], result[i], "CPU result should be consistent!");
            }

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                for (int i = 0; i < n; ++i) {
                    result[i] = as[i];
                    if (i) {
                        result[i] += result[i - 1];
                    }
                }
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << n * 1e-6 / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            gpu::WorkSize ws(workGroupSize, n);

            ocl::Kernel kernel(prefix_sum_kernel, prefix_sum_kernel_length, "cumsum_naive", defines.str());

            gpu::gpu_mem_32u as_gpu;
            gpu::gpu_mem_32u bs_gpu;
            as_gpu.resizeN(n);
            bs_gpu.resizeN(n);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                t.restart();
                for (int ready_len = 1; ready_len < n; ready_len *= 2) {
                    kernel.exec(ws, as_gpu, bs_gpu, n, ready_len);
                    std::swap(as_gpu, bs_gpu);
                }
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << n * 1e-6 / t.lapAvg() << " millions/s" << std::endl;

            as_gpu.readN(result.data(), n);
            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(reference_result[i], result[i], "GPU result");
            }
        }
    }
}
