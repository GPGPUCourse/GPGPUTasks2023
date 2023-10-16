#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"


template<typename T>
void raiseFail(const T &a, const T &b, int index, std::string message, std::string filename, int line)
{
	if (a != b) {
		std::cerr << message << '#' << index << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
		throw std::runtime_error(message);
	}
}

#define EXPECT_THE_SAME(a, b, index, message) raiseFail(a, b, index, message, __FILE__, __LINE__)


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
					EXPECT_THE_SAME(reference_result[i], result[i], i, "CPU result should be consistent!");
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
			std::cout << "CPU: " << n * 1e-6 / t.lapAvg() << " millions/s" << std::endl;
		}

        gpu::gpu_mem_32u as_gpu;
        as_gpu.resizeN(n);
        std::vector<unsigned int> res(n, 0);

        {
            ocl::Kernel prefix_sum_scan(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum_scan");
            prefix_sum_scan.compile();

            unsigned int workGroupSize = 128;
            unsigned int global_work_size = (n / 2 + workGroupSize - 1) / workGroupSize * workGroupSize;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);

                t.restart(); // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

                for (int width = 2; width < 2 * n; width <<= 1) {
                    prefix_sum_scan.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                         as_gpu, n, width);
                }
                t.nextLap();
            }
            std::cout << "A sum scan algorithm" << std::endl;
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << n * 1e-6 / t.lapAvg() << " millions/s" << std::endl;

            as_gpu.readN(res.data(), n);
        }

        // Проверяем корректность результатов
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(res[i], reference_result[i], i, "GPU results should be equal to CPU results!");
        }


        {
            ocl::Kernel prefix_sum_map(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum_map");
            prefix_sum_map.compile();

            ocl::Kernel prefix_sum_reduce(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum_reduce");
            prefix_sum_reduce.compile();

            as.push_back(0);
            as_gpu.resizeN(n + 1);
            unsigned int workGroupSize = 128;
            unsigned int global_work_size = 0;

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n + 1); // нужно занулить последний элемент для многократного исполнения ядра
                t.restart(); // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных

                int width;
                for (width = 2; width < 2 * n; width <<= 1) {
                    global_work_size = (n / width + workGroupSize - 1) / workGroupSize * workGroupSize;
                    global_work_size = std::max(global_work_size, 1U);
                    prefix_sum_map.exec(gpu::WorkSize(std::min(workGroupSize, global_work_size), global_work_size),
                                        as_gpu, n, width);
                }

                for (; width > 1; width >>= 1) {
                    global_work_size = (n / width + workGroupSize - 1) / workGroupSize * workGroupSize;
                    global_work_size = std::max(global_work_size, 1U);
                    prefix_sum_reduce.exec(gpu::WorkSize(std::min(workGroupSize, global_work_size), global_work_size),
                                        as_gpu, n, width);
                }
                t.nextLap();
            }
            std::cout << "A sum parallel scan" << std::endl;
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << n * 1e-6 / t.lapAvg() << " millions/s" << std::endl;

            as_gpu.readN(res.data(), n, 1);
        }

        // Проверяем корректность результатов
        for (int i = 0; i < n; ++i) {
            EXPECT_THE_SAME(res[i], reference_result[i], i, "GPU results should be equal to CPU results!");
        }
    }
}
