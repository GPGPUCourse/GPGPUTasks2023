#include <cmath>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"
#include "libgpu/utils.h"
#include "libgpu/work_size.h"


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
                    bs[i] += bs[i - 1];
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
                        result[i] += result[i - 1];
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
                        result[i] += result[i - 1];
                    }
                }
                t.nextLap();
            }
            std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "CPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }

		std::vector<uint32_t> res(n);
	    gpu::gpu_mem_32u as_gpu, res_gpu;
    	as_gpu.resizeN(n);
		res_gpu.resizeN(n);

		{
			ocl::Kernel sweep_up(prefix_sum_kernel, prefix_sum_kernel_length, "sweep_up");
			ocl::Kernel sweep_down(prefix_sum_kernel, prefix_sum_kernel_length, "sweep_down");
			ocl::Kernel set_zero(prefix_sum_kernel, prefix_sum_kernel_length, "set_zero");
			ocl::Kernel shift_left(prefix_sum_kernel, prefix_sum_kernel_length, "shift_left");
			sweep_up.compile();
			sweep_down.compile();
			set_zero.compile();
			shift_left.compile();
			timer t;
			for (int iter = 0; iter < benchmarkingIters; ++iter) {
				as_gpu.writeN(as.data(), n);
				res_gpu.writeN(res.data(), n);

				t.restart();

				uint32_t workGroupSize = 2;

				for (int offset = 1; offset < n; offset <<= 1) {
					uint32_t globalWorkSize = gpu::divup(n / (offset << 1), workGroupSize) * workGroupSize;
					sweep_up.exec(gpu::WorkSize(workGroupSize, globalWorkSize), as_gpu, n, offset);
				}

				set_zero.exec(gpu::WorkSize(workGroupSize, 1), as_gpu, n);

				for (int offset = n >> 1; offset > 0; offset >>=1) {
					uint32_t globalWorkSize = gpu::divup(n / offset, workGroupSize) * workGroupSize;
					sweep_down.exec(gpu::WorkSize(workGroupSize, globalWorkSize), as_gpu, n, offset);
				}

				shift_left.exec(gpu::WorkSize(workGroupSize, gpu::divup(n, workGroupSize) * workGroupSize), as_gpu, res_gpu, n);

				t.nextLap();
			}
			std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
			std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;

			res_gpu.readN(res.data(), n);
			res[n - 1] += as[n - 1];
		}

		for (int i = 0; i < n; ++i) {
			EXPECT_THE_SAME(res[i], reference_result[i], "GPU results should be equal to CPU results!");
		}
    }
}
