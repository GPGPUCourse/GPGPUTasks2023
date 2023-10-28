#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"
#include "cl_defines.h"

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

#include <sstream>

static std::string get_defines() {
	std::stringstream str;
	str << "-D " << "RADIX_BITS=" << RADIX_BITS << " ";
	str << "-D " << "WORKGROUP_SIZE=" << WORKGROUP_SIZE << " ";
	str << "-D " << "WORK_PER_THREAD=" << WORK_PER_THREAD << " ";
	str << "-D " << "WITH_LOCAL_SORT=" << WITH_LOCAL_SORT << " ";
	str << "-D " << "TRANSPOSE_WORKGROUP_SIZE=" << TRANSPOSE_WORKGROUP_SIZE << " ";
	return str.str();
}

static ocl::Kernel get_kernel(std::string kernel_name) {
	ocl::Kernel kern(radix_kernel, radix_kernel_length, kernel_name, get_defines());
	kern.compile();
	return std::move(kern);
}

// is it too much?
class scanSingleton
{
private:
	scanSingleton() {
		reduce_kernel = get_kernel("scan_reduce_global");
		down_sweep_kernel = get_kernel("scan_down_sweep_global");
	}
	scanSingleton( const scanSingleton&);
	scanSingleton& operator=( scanSingleton& );
	ocl::Kernel reduce_kernel;
	ocl::Kernel down_sweep_kernel;
public:
	static scanSingleton& getInstance() {
		static scanSingleton instance;
		return instance;
	}

	void scan(const gpu::gpu_mem_32u &buffer,
	unsigned int n) {
		int binsPerChunk = 1 << RADIX_BITS;
		int numChunks = n / WORKGROUP_SIZE;
		int buffer_size = binsPerChunk * numChunks;
		unsigned int offset;
		for (offset = 1; offset * 2 <= buffer_size; offset <<= 1) {
			reduce_kernel.exec(gpu::WorkSize(WORKGROUP_SIZE, (buffer_size + offset - 1) / offset),
							   buffer,
							   buffer_size,
							   offset);
		}

		for (offset = offset >> 1; offset >= 1; offset >>= 1) {
			down_sweep_kernel.exec(gpu::WorkSize(WORKGROUP_SIZE,(buffer_size + offset - 1) / offset),
								   buffer,
								   buffer_size,
								   offset);
		}
	}
};

//// uses precompiled kernels
//void scan(ocl::Kernel &reduce_kernel,
//		  ocl::Kernel &down_sweep_kernel,
//		  gpu::gpu_mem_32u buffer,
//		  unsigned int n,
//		  int bit_offset) {
//	int binsPerChunk = 1 << RADIX_BITS;
//	int numChunks = n / WORK_PER_THREAD;
//	unsigned int offset;
//	for (offset = 1; offset * 2 <= n; offset <<= 1) {
//		reduce_kernel.exec(gpu::WorkSize(),
//						 as_gpu,
//						 n,
//						 offset);
//	}
//
//	for (offset = offset >> 1; offset >= 1; offset >>= 1) {
//		down_sweep_kernel.exec(gpu::WorkSize(workGroupSize, (n + offset - 1) / offset),
//							 as_gpu,
//							 n,
//							 offset);
//	}
//
//}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

//    int benchmarkingIters = 10;
    int benchmarkingIters = 1;
//    unsigned int n = 32 * 1024 * 1024;
//    unsigned int n = 1024;
    unsigned int n = 256;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
//        as[i] = (unsigned int) r.next(0, 3);
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
//    {
//        timer t;
//        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
//            std::sort(cpu_sorted.begin(), cpu_sorted.end());
//            t.nextLap();
//        }
//        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
//        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
//    }

    gpu::gpu_mem_32u as_gpu;
    gpu::gpu_mem_32u buffer_gpu;
    gpu::gpu_mem_32u buffer_copy_gpu;
    as_gpu.resizeN(n);
	unsigned int binsPerChunk = 1 << RADIX_BITS;
	unsigned int numChunks = (n  + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;
	buffer_gpu.resizeN(numChunks * binsPerChunk);
	buffer_copy_gpu.resizeN(numChunks * binsPerChunk);

	unsigned int work_size = numChunks * WORKGROUP_SIZE;

    {
        ocl::Kernel radix = get_kernel("radix");
        ocl::Kernel count = get_kernel("count");
        ocl::Kernel transpose = get_kernel("matrix_transpose");
		// precompile `scan` kernels
		scanSingleton::getInstance();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
			for (int bit_offset = 0; bit_offset < 32; bit_offset += RADIX_BITS) {
//			for (int bit_offset = 0; bit_offset <= 0; bit_offset += RADIX_BITS) {
				// count into bins
				count.exec(gpu::WorkSize(WORKGROUP_SIZE, work_size), as_gpu, buffer_gpu, n, bit_offset);
				// transpose to increase coalesced access
				transpose.exec(gpu::WorkSize(TRANSPOSE_WORKGROUP_SIZE,
											 TRANSPOSE_WORKGROUP_SIZE,
											 (binsPerChunk + TRANSPOSE_WORKGROUP_SIZE - 1) / TRANSPOSE_WORKGROUP_SIZE * TRANSPOSE_WORKGROUP_SIZE,
											 (numChunks + TRANSPOSE_WORKGROUP_SIZE - 1) / TRANSPOSE_WORKGROUP_SIZE * TRANSPOSE_WORKGROUP_SIZE), buffer_gpu, buffer_copy_gpu, binsPerChunk, numChunks);
				std::swap(buffer_gpu, buffer_copy_gpu);

				buffer_gpu.readN(as.data(), binsPerChunk * numChunks);
				for (int i = 0; i < binsPerChunk; ++i) {
					for (int j = 0; j < numChunks; ++j) {
						std::cout << as[i * numChunks + j] << " ";
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
				// prefix sum (aka scan) on bins
				scanSingleton::getInstance().scan(buffer_gpu, n);

				buffer_gpu.readN(as.data(), binsPerChunk * numChunks);
				for (int i = 0; i < binsPerChunk; ++i) {
					for (int j = 0; j < numChunks; ++j) {
						std::cout << as[i * numChunks + j] << " ";
					}
					std::cout << std::endl;
				}
				std::cout << std::endl;
				// sort using bin info
				radix.exec(gpu::WorkSize(WORKGROUP_SIZE, work_size), as_gpu, buffer_gpu, n, bit_offset);
			}

			t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;

        as_gpu.readN(as.data(), n);

		// Проверяем корректность результатов
		for (int i = 0; i < n; ++i) {
			EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
		}
	}

    return 0;
}

