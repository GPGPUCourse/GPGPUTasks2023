#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
	if (a != b) {
		std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
		throw std::runtime_error(message);
	}
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


int main(int argc, char **argv)
{
	int benchmarkingIters = 10;
	unsigned int max_n = (1 << 24);

    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

	for (unsigned int n = 4096; n <= max_n; n *= 4) {
		std::cout << "______________________________________________" << std::endl;
		unsigned int values_range = std::min<unsigned int>(1023, std::numeric_limits<int>::max() / n);
		std::cout << "n=" << n << " values in range: [" << 0 << "; " << values_range << "]" << std::endl;

		std::vector<unsigned int> as(n, 0);
		FastRandom r(n);
		for (int i = 0; i < n; ++i) {
			as[i] = r.next(0, values_range);
		}

		std::vector<unsigned int> bs(n, 0);
		{
			for (int i = 0; i < n; ++i) {
				bs[i] = as[i];
				if (i) {
					bs[i] += bs[i-1];
				}
			}
		}
		const std::vector<unsigned int> reference_result = bs;

		{
			{
				std::vector<unsigned int> result(n);
				for (int i = 0; i < n; ++i) {
					result[i] = as[i];
					if (i) {
						result[i] += result[i-1];
					}
				}
				for (int i = 0; i < n; ++i) {
					EXPECT_THE_SAME(reference_result[i], result[i], "CPU result should be consistent!");
				}
			}

			std::vector<unsigned int> result(n);
			timer t;
			for (int iter = 0; iter < benchmarkingIters; ++iter) {
				for (int i = 0; i < n; ++i) {
					result[i] = as[i];
					if (i) {
						result[i] += result[i-1];
					}
				}
				t.nextLap();
			}
			std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
			std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
		}

		{
        // TODO: implement on OpenCL
        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        as_gpu.writeN(as.data(), n);

        ocl::Kernel prefix(prefix_sum_kernel, prefix_sum_kernel_length, "prefix");
        ocl::Kernel reduce(prefix_sum_kernel, prefix_sum_kernel_length, "reduce");
        prefix.compile();
        reduce.compile();

        std::vector<unsigned int> result(n, 0);
        gpu::gpu_mem_32u result_gpu;
        result_gpu.resizeN(n);
        result_gpu.writeN(result.data(), n);

        for (unsigned k = 0; (1u << k) <= n; ++k) {
            reduce.exec(gpu::WorkSize(256, n), as_gpu, k);
            prefix.exec(gpu::WorkSize(256, n), as_gpu, result_gpu, k);
        }
        result_gpu.readN(result.data(), n);
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(reference_result[i], result[i], "GPU result should be equal to CPU!");
        }

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);

            t.restart();
            for (unsigned k = 0; (1u << k) <= n; ++k) {
                reduce.exec(gpu::WorkSize(256, n), as_gpu, k);
                prefix.exec(gpu::WorkSize(256, n), as_gpu, result_gpu, k);
            }
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

        }
    }
}
