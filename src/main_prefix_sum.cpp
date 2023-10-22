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

#define WORKGROUP_SIZE 128

int main(int argc, char **argv)
{
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

	int benchmarkingIters = 10;
	unsigned int max_n = (1 << 24);

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
            int N = 1;
            while (N < n || N < WORKGROUP_SIZE) N *= 2;
            as.resize(N, INFINITY);

            gpu::gpu_mem_32u as_gpu, bs_gpu;
            as_gpu.resizeN(N);
            bs_gpu.resizeN(N);

            as_gpu.writeN(as.data(), N);

            ocl::Kernel sums1(prefix_sum_kernel, prefix_sum_kernel_length, "sums1", "-D WORKGROUP_SIZE=" + to_string(WORKGROUP_SIZE));
            sums1.compile();
            ocl::Kernel sums2(prefix_sum_kernel, prefix_sum_kernel_length, "sums2", "-D WORKGROUP_SIZE=" + to_string(WORKGROUP_SIZE));
            sums2.compile();

            {
                for (int len = 1; len <= N; len <<= 1)
                {
                    int workGroupSize = WORKGROUP_SIZE;
                    int all_work = (N / len + workGroupSize - 1) / workGroupSize * workGroupSize;
                    sums1.exec(gpu::WorkSize(workGroupSize, all_work), as_gpu, bs_gpu, len, N);
                }
                int workGroupSize = WORKGROUP_SIZE;
                int all_work = (N + workGroupSize - 1) / workGroupSize * workGroupSize;
                sums2.exec(gpu::WorkSize(workGroupSize, all_work), as_gpu, bs_gpu, N);

                std::vector<unsigned int> result(n);
                as_gpu.readN(result.data(), n);
                for (int i = 0; i < n; ++i) {
                    EXPECT_THE_SAME(reference_result[i], result[i], "GPU result should be consistent!");
                }
            }

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter)
            {
                as_gpu.writeN(as.data(), N);
                t.restart();

                for (int len = 1; len <= N; len <<= 1)
                {
                    int workGroupSize = WORKGROUP_SIZE;
                    int all_work = (N / len + workGroupSize - 1) / workGroupSize * workGroupSize;
                    sums1.exec(gpu::WorkSize(workGroupSize, all_work), as_gpu, bs_gpu, len, N);
                }
                int workGroupSize = WORKGROUP_SIZE;
                int all_work = (N + workGroupSize - 1) / workGroupSize * workGroupSize;
                sums2.exec(gpu::WorkSize(workGroupSize, all_work), as_gpu, bs_gpu, N);

                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
		}
	}
}
