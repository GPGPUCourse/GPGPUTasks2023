#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#include <sstream>

#include "CL/cl.h"

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

    constexpr int benchmarkingIters = 10;

    constexpr cl_uint workGroupSize = 64;
    std::ostringstream defines;
    defines << "-DWORK_GROUP_SIZE=" << workGroupSize;

    for (cl_uint log2n = 12; log2n <= 24; log2n += 2) {
        cl_uint n = 1 << log2n;
        std::cout << "______________________________________________" << std::endl;
        cl_uint values_range = std::min<cl_uint>(1023, std::numeric_limits<int>::max() / n);
        std::cout << "n = 2^" << log2n << " = " << n << " values in range: [" << 0 << "; " << values_range << "]\n";

        std::vector<cl_uint> as(n, 0);
        FastRandom r(n);
        for (int i = 0; i < n; ++i) {
            as[i] = r.next(0, values_range);
        }

        std::vector<cl_uint> result(n, 0);
        for (int i = 0; i < n; ++i) {
            result[i] = as[i];
            if (i) {
                result[i] += result[i - 1];
            }
        }
        const std::vector<cl_uint> reference_result = result;

        {
            for (int i = 0; i < n; ++i) {
                result[i] = as[i];
                if (i) {
                    result[i] += result[i - 1];
                }
            }
            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(reference_result[i], result[i], "CPU result should be consistent!");
            }

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
            std::cout << "CPU: " << n * 1e-6 / t.lapAvg() << " millions/s" << std::endl;
        }

        {
            gpu::WorkSize ws(workGroupSize, n);

            ocl::Kernel kernel(prefix_sum_kernel, prefix_sum_kernel_length, "cumsum_naive", defines.str());

            gpu::gpu_mem_32u as_gpu;
            gpu::gpu_mem_32u bs_gpu;
            as_gpu.resizeN(n);
            bs_gpu.resizeN(n);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                t.restart();
                for (cl_uint step = 1; step < n; step *= 2) {
                    kernel.exec(ws, as_gpu, bs_gpu, n, step);
                    std::swap(as_gpu, bs_gpu);
                }
                t.nextLap();
            }
            std::cout << "GPU naive: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU naive: " << n * 1e-6 / t.lapAvg() << " millions/s" << std::endl;

            as_gpu.readN(result.data(), n);
            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(reference_result[i], result[i], "GPU result should be the same as CPU");
            }
        }

        {
            ocl::Kernel sweep_up(prefix_sum_kernel, prefix_sum_kernel_length, "cumsum_sweep_up", defines.str());
            ocl::Kernel sweep_down(prefix_sum_kernel, prefix_sum_kernel_length, "cumsum_sweep_down", defines.str());
            ocl::Kernel shift_left(prefix_sum_kernel, prefix_sum_kernel_length, "shift_left", defines.str());

            gpu::gpu_mem_32u as_gpu;
            gpu::gpu_mem_32u bs_gpu;
            as_gpu.resizeN(n);
            bs_gpu.resizeN(n);

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);
                t.restart();

                for (cl_uint step = 1; step < n; step *= 2) {
                    uint len = n / 2 / step;
                    gpu::WorkSize ws(workGroupSize, len);
                    sweep_up.exec(ws, as_gpu, len, step);
                }

                // Можно было бы добавить total в конец вектора на процессоре,
                // но предполагаем, что задача --- получить кумулятивную сумму
                // на видеокарте (иначе бы не имело смысла пересылать данные вообще),
                // поэтому будем делать смещение влево на 1 шаг и дополнение
                // полной суммой.
                // В результате этих неявных требований
                // получается существенное падение производительности
                // на GTX 1060, с 1400 млн/с до 1200 млн/с
                cl_uint total = 0;
                as_gpu.readN(&total, 1, n - 1);

                // Нам не дают clEnqueueFillBuffer :(
                cl_uint zero = 0;
                context.cl()->writeBuffer(as_gpu.clmem(), CL_TRUE, (n - 1) * sizeof(cl_uint), sizeof(cl_uint), &zero);

                for (cl_uint step = n / 2; step > 0; step /= 2) {
                    uint len = n / 2 / step;
                    gpu::WorkSize ws(workGroupSize, len);
                    sweep_down.exec(ws, as_gpu, len, step);
                }

                context.cl()->copyBuffer(as_gpu.clmem(), bs_gpu.clmem(), sizeof(cl_uint), 0, (n - 1) * sizeof(cl_uint));
                context.cl()->writeBuffer(bs_gpu.clmem(), CL_TRUE, (n - 1) * sizeof(cl_uint), sizeof(cl_uint), &total);
                t.nextLap();
            }
            std::cout << "GPU tree: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
            std::cout << "GPU tree: " << n * 1e-6 / t.lapAvg() << " millions/s" << std::endl;

            bs_gpu.readN(result.data(), n);
            for (int i = 0; i < n; ++i) {
                EXPECT_THE_SAME(reference_result[i], result[i], "GPU result");
            }
        }
    }
}
