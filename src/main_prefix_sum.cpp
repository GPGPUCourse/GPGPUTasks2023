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
			bs.assign(n, 0);

			gpu::gpu_mem_32u as_gpu, ss_gpu;
    		as_gpu.resizeN(n);
    		ss_gpu.resizeN(n);

			ocl::Kernel prefix_sum(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum");
       		prefix_sum.compile();

			ocl::Kernel accum(prefix_sum_kernel, prefix_sum_kernel_length, "accum");
       		accum.compile();

			timer t;
			for (int iter = 0; iter < 1; ++iter) {
				as_gpu.writeN(bs.data(), n);
				ss_gpu.writeN(as.data(), n);

				t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
				
				const int work_group_size = 128;
				prefix_sum.exec(
					gpu::WorkSize(work_group_size, n), 
					as_gpu, 
					ss_gpu,
					1
				);
				for(int i=2;i<=n;i<<=1) {
					accum.exec(
						gpu::WorkSize(work_group_size, ((n/i)+work_group_size-1)/work_group_size*work_group_size), 
						ss_gpu, 
						i,
						n
					);
					prefix_sum.exec(
						gpu::WorkSize(work_group_size, n), 
						as_gpu, 
						ss_gpu,
						i
					);
				}
				
				t.nextLap();
			}

			std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
			std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

			as_gpu.readN(as.data(), n);
		}

		// Проверяем корректность результатов
		for (int i = 0; i < n; ++i) {
			EXPECT_THE_SAME(as[i], reference_result[i], "GPU results should be equal to CPU results!" + to_string(i));
		}
	}
}
