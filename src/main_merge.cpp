#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

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

    int benchmarkingIters = 1;
//    unsigned int n = 32 * 1024 * 1024;
    unsigned int n = 1024;
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
    gpu::gpu_mem_32f bs_gpu;
    gpu::gpu_mem_32u intervals_gpu;
    as_gpu.resizeN(n);
    bs_gpu.resizeN(n);
	unsigned int workGroupSize = 16;
	unsigned int partitionsWorkSize = (n + workGroupSize - 1) / (workGroupSize * 2);
	intervals_gpu.resizeN(partitionsWorkSize * 4); // i'm too lazy to count all of it
#define WORK_PER_THREAD 2
    {
        ocl::Kernel merge_small(merge_kernel, merge_kernel_length, "merge_small");
        ocl::Kernel merge(merge_kernel, merge_kernel_length, "merge");
        ocl::Kernel find_partitions(merge_kernel, merge_kernel_length, "find_partitions");
		merge_small.compile();
		merge.compile();
		find_partitions.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
//			for (int i = 0; i < workGroupSize * 2; ++i) {
//				std::cout << as[i] << " ";
//			}
//			std::cout << ";\n";
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных
            unsigned int global_work_size = (n + (workGroupSize * 2) - 1) / (workGroupSize * 2) * workGroupSize;
			merge_small.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, n);
			as_gpu.readN(as.data(), n);
			for (int i = 0; i < n; ++i) {
				std::cout << as[i] << " ";
				if (i % workGroupSize == workGroupSize - 1)
					std::cout << ";\n";
			}
			std::cout << std::endl;
			std::cout << std::endl;
			for (int sort_size = workGroupSize * 2; sort_size <= n / 2; sort_size <<= 1)
//			for (int sort_size = workGroupSize * 2; sort_size <= workGroupSize * 2; sort_size <<= 1)
			{
				find_partitions.exec(gpu::WorkSize(workGroupSize, partitionsWorkSize), as_gpu, intervals_gpu, n, sort_size);
				std::vector<unsigned int> parts(partitionsWorkSize * 4);
				intervals_gpu.readN(parts.data(), partitionsWorkSize * 4);
				for (int i = 0; i < partitionsWorkSize * 4; ++i) {
					std::cout << parts[i] << " ";
					if (i % partitionsWorkSize == partitionsWorkSize - 1)
						std::cout << ";\n";
				}
				std::cout << std::endl;
				merge.exec(gpu::WorkSize(workGroupSize, partitionsWorkSize), as_gpu, bs_gpu, intervals_gpu, n);
				std::swap(as_gpu, bs_gpu);

				as_gpu.readN(as.data(), sort_size * 2);
				for (int i = 0; i < sort_size * 2; ++i) {
					std::cout << as[i] << " ";
					if (i % workGroupSize == workGroupSize - 1)
						std::cout << ";\n";
				}
				std::cout << std::endl;
			}
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
		as_gpu.readN(as.data(), n);
    }
	for (int i = 0; i < n; ++i) {
		std::cout << as[i] << " ";
		if (i % workGroupSize == workGroupSize - 1)
			std::cout << ";\n";
	}
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
