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

#define WORK_GROUP_SIZE 128

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
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
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32f as_gpu;
    as_gpu.resizeN(n);

	std::vector<float> gpu_sorted(n);
    {
        ocl::Kernel bitonic(bitonic_kernel, bitonic_kernel_length, "bitonic");
        bitonic.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

			for (unsigned int i = 1; i <= n >> 1; i <<= 1) {
				for (unsigned int j = i; j >= 1; j >>= 1) {
					bitonic.exec(gpu::WorkSize(WORK_GROUP_SIZE, n), as_gpu, n, i, j);
				}
			}
			t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(gpu_sorted.data(), n);
    }

    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(gpu_sorted[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

	{
		ocl::Kernel bitonic_fast(bitonic_kernel, bitonic_kernel_length, "bitonic_fast");
		bitonic_fast.compile();

		int max_pow_of_2_less_than_n = 32;
		while (!(1 << (--max_pow_of_2_less_than_n) & n));

		timer t;
		for (int iter = 0; iter < benchmarkingIters; ++iter) {
			as_gpu.writeN(as.data(), n);

			t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

			for (int i = 0; i < max_pow_of_2_less_than_n; ++i) {
				for (int j = i; j >= 0; --j) {
					bitonic_fast.exec(gpu::WorkSize(WORK_GROUP_SIZE, n), as_gpu, n, (unsigned int)i, (unsigned int)j);
				}
			}
			t.nextLap();
		}
		std::cout << "GPU \"fast\": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
		std::cout << "GPU \"fast\": " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

		as_gpu.readN(gpu_sorted.data(), n);
	}

	// Проверяем корректность результатов
	for (int i = 0; i < n; ++i) {
		EXPECT_THE_SAME(gpu_sorted[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
	}

    return 0;
}
