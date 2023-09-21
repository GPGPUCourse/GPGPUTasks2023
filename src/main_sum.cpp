#include "cl/sum_cl.h"
#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <cassert>
#include <fstream>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

std::string to_string() {
    return "";
}

template<typename T, typename... TAIL>
std::string to_string(const T &t, TAIL... tail) {
    std::stringstream ss;
    ss << t << to_string(tail...);
    return ss.str();
}

template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)


double execKernel(ocl::Kernel &kernel, gpu::gpu_mem_32u const &aGpu, gpu::gpu_mem_32u &resGPU, uint workGroupSize, uint globalWorkSize, uint n,
                  uint expectedSum, uint benchmarkingIters, std::string kernelName) {
    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        uint actualSum = 0;
        resGPU.writeN(&actualSum, 1);
        kernel.exec(gpu::WorkSize(workGroupSize, globalWorkSize), aGpu, resGPU, n);
        resGPU.readN(&actualSum, 1);
        EXPECT_THE_SAME(expectedSum, actualSum, "GPU (" + kernelName + ") result should be consistent!");
        t.nextLap();
    }

    std::cout << "GPU (" << kernelName << "):     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    double ops = (n / 1000.0 / 1000.0) / t.lapAvg();
    std::cout << "GPU (" << kernelName << "):     " << ops << " millions/s" << std::endl;
    return ops;
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);
    std::string outputFolder = to_string("temp/", argv[1], "-");
    int benchmarkingIters = 20;

    unsigned int reference_sum = 0;
    unsigned int n = 100 * 1000 * 1000;
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
            EXPECT_THE_SAME(reference_sum, sum, "CPU result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:     " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
#pragma omp parallel for reduction(+ : sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n / 1000.0 / 1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        // TODO: implement on OpenCL
        gpu::Context context;
        context.init(device.device_id_opencl);
        context.activate();

        gpu::gpu_mem_32u aGpu;
        gpu::gpu_mem_32u resGPU;
        unsigned int workGroupSize = 64;
        unsigned int globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        aGpu.resizeN(globalWorkSize);
        as.resize(globalWorkSize);
        resGPU.resizeN(1);
        aGpu.writeN(as.data(), globalWorkSize);
        {
            ocl::Kernel atomicAddSum(sum_kernel, sum_kernel_length, "atomicAddSum");
            execKernel(atomicAddSum, aGpu, resGPU, workGroupSize, globalWorkSize, as.size(), reference_sum, benchmarkingIters, "atomicAddSum");
        }

        {
            uint valuesPerItem = 64;
            uint workSize = (globalWorkSize + valuesPerItem - 1) / valuesPerItem;
            ocl::Kernel loopSum(sum_kernel, sum_kernel_length, "loopSum", to_string("-DVALUES_PER_WORKITEM=", valuesPerItem));
            execKernel(loopSum, aGpu, resGPU, workGroupSize, globalWorkSize / valuesPerItem, as.size(), reference_sum, benchmarkingIters,
                       to_string("loopSum (", valuesPerItem, ")"));
        }

        {
            uint valuesPerItem = 64;
            uint workSize = (globalWorkSize + valuesPerItem - 1) / valuesPerItem;
            ocl::Kernel loopCoalesedSum(sum_kernel, sum_kernel_length, "loopCoalesedSum", to_string("-DVALUES_PER_WORKITEM=", valuesPerItem));
            execKernel(loopCoalesedSum, aGpu, resGPU, workGroupSize, workSize, as.size(), reference_sum, benchmarkingIters,
                       "loopCoalesedSum (" + std::to_string(valuesPerItem) + ")");
        }


        {
            ocl::Kernel localMemSum(sum_kernel, sum_kernel_length, "localMemSum");
            execKernel(localMemSum, aGpu, resGPU, workGroupSize, globalWorkSize, as.size(), reference_sum, benchmarkingIters, "localMemSum");
        }

        {
            ocl::Kernel treeSum(sum_kernel, sum_kernel_length, "treeSum");
            execKernel(treeSum, aGpu, resGPU, workGroupSize, globalWorkSize, as.size(), reference_sum, benchmarkingIters, "treeSum");
        }

        {
            std::ofstream fout(outputFolder + "loopSum");
            std::cout << "diff loop sizes:" << std::endl;
            for (int valuesPerItem : std::vector<int>{1, 2, 4, 6, 8, 12, 16, 32, 128, 512, 1024, 4096, 8192}) {
                int workSize = (globalWorkSize + valuesPerItem - 1) / valuesPerItem;
                ocl::Kernel loopSum(sum_kernel, sum_kernel_length, "loopSum", to_string("-DVALUES_PER_WORKITEM=", valuesPerItem));
                double ops = execKernel(loopSum, aGpu, resGPU, workGroupSize, workSize, as.size(), reference_sum, benchmarkingIters,
                                        to_string("loopSum (", valuesPerItem, ")"));
                fout << valuesPerItem << ' ' << ops << std::endl;
            }
        }

        {
            std::ofstream fout(outputFolder + "loopCoalesedSum");
            std::cout << "diff loop coalesed sizes:" << std::endl;
            for (int valuesPerItem : std::vector<int>{1, 2, 4, 6, 8, 12, 16, 32, 128, 512, 1024, 4096, 8192}) {
                int workSize = (globalWorkSize + valuesPerItem - 1) / valuesPerItem;
                ocl::Kernel loopCoalesedSum(sum_kernel, sum_kernel_length, "loopCoalesedSum", to_string("-DVALUES_PER_WORKITEM=", valuesPerItem));
                double ops = execKernel(loopCoalesedSum, aGpu, resGPU, workGroupSize, workSize, as.size(), reference_sum, benchmarkingIters,
                                        to_string("loopCoalesedSum (", valuesPerItem, ")"));
                fout << valuesPerItem << ' ' << ops << std::endl;
            }
        }

        {
            std::ofstream fout(outputFolder + "localMemSum");
            std::cout << "diff localMem sizes:" << std::endl;
            for (int workGroupSize : std::vector<int>{1, 2, 4, 6, 8, 12, 16, 32, 64, 128, 192, 256}) {
                ocl::Kernel localMemSum(sum_kernel, sum_kernel_length, "localMemSum", to_string("-DWORKGROUP_SIZE=", workGroupSize));
                double ops = execKernel(localMemSum, aGpu, resGPU, workGroupSize, globalWorkSize, as.size(), reference_sum, benchmarkingIters,
                                        to_string("localMemSum (", workGroupSize, ")"));
                fout << workGroupSize << ' ' << ops << std::endl;
            }
        }

        {
            std::ofstream fout(outputFolder + "treeSum");
            std::cout << "diff tree sizes:" << std::endl;
            for (int workGroupSize : std::vector<int>{1, 2, 4, 8, 16, 32, 64, 128, 256}) {
                ocl::Kernel localMemSum(sum_kernel, sum_kernel_length, "treeSum", to_string("-DWORKGROUP_SIZE=", workGroupSize));
                double ops = execKernel(localMemSum, aGpu, resGPU, workGroupSize, globalWorkSize, as.size(), reference_sum, benchmarkingIters,
                                        to_string("treeSum (", workGroupSize, ")"));
                fout << workGroupSize << ' ' << ops << std::endl;
            }
        }
    }
}
