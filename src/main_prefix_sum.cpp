#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/prefix_sum_cl.h"


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

        {
            ocl::Kernel prefix_sum(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum");
            prefix_sum.compile();


            const std::vector<unsigned int> zeros(n, 0);
			unsigned int workGroupSize = 128;
            unsigned int global_work_size =
                    ((n / 2) / workGroupSize + ((n / 2) % workGroupSize > 0 ? 1 : 0)) * workGroupSize;
            {
                gpu::gpu_mem_32u as_gpu;
                as_gpu.resizeN(n);
            	as_gpu.writeN(as.data(), n);

                gpu::gpu_mem_32u result_gpu;
                result_gpu.resizeN(n);
                result_gpu.writeN(zeros.data(), n);
                
                for (unsigned int cur_block_size = 1; cur_block_size < n; cur_block_size <<= 1) {
                    prefix_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, result_gpu, n,
                                    cur_block_size);
                }

				std::vector<unsigned int> result(n);
                
				as_gpu.readN(result.data(), n);


				result_gpu.readN(result.data(), n);
                for (int i = 0; i < 32; ++i) {
                    EXPECT_THE_SAME(reference_result[i], result[i], "CPU result should be consistent!");
                }
            }

            std::vector<unsigned int> result(n);
            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
				gpu::gpu_mem_32u as_gpu;
                as_gpu.resizeN(n);
            	as_gpu.writeN(as.data(), n);

                gpu::gpu_mem_32u result_gpu;
                result_gpu.resizeN(n);
                result_gpu.writeN(zeros.data(), n);
                t.restart();
                for (unsigned int cur_block_size = 1; cur_block_size < n; cur_block_size <<= 1) {
                    prefix_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, result_gpu, n,
                                    cur_block_size);
                }
                prefix_sum.exec(gpu::WorkSize(workGroupSize, global_work_size), as_gpu, result_gpu, n);
                t.nextLap();
            }
            std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        }
    }
}
