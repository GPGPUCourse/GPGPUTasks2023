#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include "libgpu/work_size.h"

#include "utils.h"

int main(int argc, char **argv) {
    unsigned int benchmarkingIters = 10;
    unsigned int n = 100 * 1000 * 1000;

    unsigned int reference_sum = 0;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(42);
    for (int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<unsigned int>::max() / n);
        reference_sum += as[i];
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            eh::EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "summation on CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "summation on CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+ : sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            eh::EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "summation on CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "summation on CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
        std::cout << std::endl;
    }

    {
        gpu::Device device = gpu::chooseGPUDevice(argc, argv);

        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();
        std::cout << std::endl;

        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        utils::ParamsUsually paramsU{trimmed(device.name), as, benchmarkingIters, reference_sum, n};

        sum(gpu::WorkSize(workGroupSize, global_work_size), paramsU, "with global atomic add", "sum1");

        unsigned int temp = global_work_size / 128;
        sum(gpu::WorkSize(workGroupSize, temp), paramsU, "with loop", "sum2");

        sum(gpu::WorkSize(workGroupSize, temp), paramsU, "with loop and coalesced access", "sum3");

        sum(gpu::WorkSize(workGroupSize, global_work_size), paramsU, "with local memory and global thread", "sum4");

        unsigned int res_n = global_work_size / workGroupSize;
        utils::ParamsTree paramsT{trimmed(device.name), as, benchmarkingIters, reference_sum, n, res_n};
        sum(gpu::WorkSize(workGroupSize, global_work_size), paramsT, "with tree", "sum5");
    }
}
