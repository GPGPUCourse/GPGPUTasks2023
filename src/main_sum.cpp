#include "cl/sum_cl.h"
#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

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

    uint referenceSum = 0;
    uint n = 100 * 1000 * 1000;
    std::vector<uint> array(n, 0);
    FastRandom random(42);
    for (int i = 0; i < n; ++i) {
        array[i] = (uint) random.next(0, std::numeric_limits<uint>::max() / n);
        referenceSum += array[i];
    }

    std::cout << "Benchmarking Iters: " << benchmarkingIters << std::endl;

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += array[i];
            }
            EXPECT_THE_SAME(referenceSum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += array[i];
            }
            EXPECT_THE_SAME(referenceSum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::shared_device_buffer_typed<uint> arrayBuffer, resultBuffer;
        arrayBuffer.resizeN(n);
        resultBuffer.resizeN(1);

        arrayBuffer.writeN(array.data(), n);

        uint workGroupSize = 128;
        uint globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        for (int i = 1; i <= 5; ++i) {
            std::cout << "Sum " << i << std::endl;
            ocl::Kernel sumKernel(sum_kernel, sum_kernel_length, "sum" + to_string(i));
            sumKernel.compile();

            timer timer;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                uint sum = 0;
                resultBuffer.writeN(&sum, 1);

                timer.start();
                sumKernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize),
                                arrayBuffer, resultBuffer, n);
                timer.nextLap();
                timer.stop();

                resultBuffer.readN(&sum, 1);

                EXPECT_THE_SAME(referenceSum, sum, "GPU result should be consistent!");
            }
            std::cout << "    GPU " << timer.lapAvg() << "+-" << timer.lapStd() << " s" << std::endl;
            std::cout << "    GPU " << (n/1000.0/1000.0) / timer.lapAvg() << " millions/s" << std::endl;
        }
    }
}
