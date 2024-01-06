#include "../libs/gpu/libgpu/context.h"
#include "../libs/gpu/libgpu/shared_device_buffer.h"
#include "../libs/utils/libutils/fast_random.h"
#include "../libs/utils/libutils/misc.h"
#include "../libs/utils/libutils/timer.h"
#include "../libs/gpu/libgpu/work_size.h"

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

#define WORKGROUP_SIZE 256

unsigned int closedPowerOfTwo(unsigned int n) {
    unsigned int result = 1;
    while (result < n) {
        result *= 2;
    }
    return result;
}

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
            gpu::gpu_mem_32u as_gpu;
            unsigned int workn = closedPowerOfTwo(n);
            as_gpu.resizeN(workn);
            gpu::gpu_mem_32u bs_gpu;
            gpu::gpu_mem_32u result_gpu;
            bs_gpu.resizeN(workn);
            result_gpu.resizeN(workn);

            auto as_complete = std::vector<unsigned int>(workn);
            auto as_buffer = std::vector<unsigned int>(n);
            auto result = std::vector<unsigned int>(workn, 0);

            for (int i = 0; i < workn; ++i) {
                if (i < as.size())
                {
                    as_complete[i] = as[i];
                }
                else
                {
                    as_complete[i] = 0;
                }
            }

            std::cout << std::endl;

            {
                ocl::Kernel prefix_sum(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum");
                ocl::Kernel reduce(prefix_sum_kernel, prefix_sum_kernel_length, "reduce");
                prefix_sum.compile();
                reduce.compile();

                timer t;
                for (int iter = 0; iter < benchmarkingIters; ++iter) {
                    result = std::vector<unsigned int>(workn, 0);
                    as_gpu.writeN(as_complete.data(), workn);
                    result_gpu.writeN(result.data(), workn);

                    t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
                    unsigned int size = workn;
                    unsigned int offset = 0;
                    while (size > 0) {
                        prefix_sum.exec(gpu::WorkSize(WORKGROUP_SIZE, workn), as_gpu, result_gpu, workn, offset);
                        reduce.exec(gpu::WorkSize(WORKGROUP_SIZE, size), as_gpu, bs_gpu, size);
                        std::swap(as_gpu, bs_gpu);
                        size /= 2;
                        offset++;
                    }
                    t.nextLap();
                    // TODO
                }
                std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
                std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

                result_gpu.readN(result.data(), n);
            }

            // Проверяем корректность результатов
            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(result[i], reference_result[i], "GPU results should be equal to CPU results at index!" + std::to_string(i));
            }
		}
	}
}
