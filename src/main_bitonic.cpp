#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/bitonic_cl.h"

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

    constexpr int benchmarkingIters = 1;// 10;
    constexpr cl_uint n = 1024 * 1024;  // 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (cl_uint i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << n * 1e-6 / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);

    {
        constexpr cl_uint workGroupSize = 64;
        gpu::WorkSize ws(workGroupSize, (n + 1) / 2);
        std::ostringstream defines;
        defines << "-DWORK_GROUP_SIZE=" << workGroupSize;

        ocl::Kernel bitonic_global(bitonic_kernel, bitonic_kernel_length, "bitonic_global", defines.str());
        bitonic_global.compile();


        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
            for (cl_uint bitonic_len = 1; bitonic_len < n; bitonic_len *= 2) {
                for (cl_uint sorted_len = bitonic_len; sorted_len > 0; sorted_len /= 2) {
                    bitonic_global.exec(ws, as_gpu, n, bitonic_len, sorted_len);
                }
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << n * 1e-6 / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);
    }

    /* for (cl_uint i = 0; i < n; ++i) {
        std::cerr << "i = " << i << "; cpu = " << cpu_sorted[i] << "; gpu = " << as[i] << '\n';
    }// */

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }// */

    return 0;
}
