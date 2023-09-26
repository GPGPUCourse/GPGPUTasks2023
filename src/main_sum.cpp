#include <libutils/misc.h>
#include <libutils/timer.h>
#include <libutils/fast_random.h>
#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>

#include "cl/sum_cl.h"
#define WORKGROUP_SIZE 128

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

	int benchmarkingIters = 10;

    unsigned int reference_sum = 0;
    unsigned int n = 100*1000*1000;
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

        std::cout << "CPU:           " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU:           " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            unsigned int sum = 0;
            #pragma omp parallel for reduction(+:sum)
            for (int i = 0; i < n; ++i) {
                sum += as[i];
            }
            EXPECT_THE_SAME(reference_sum, sum, "CPU OpenMP result should be consistent!");
            t.nextLap();
        }

        std::cout << "CPU OMP:       " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP:       " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
    }

	// TODO: implement on OpenCL
	gpu::Context context;
	context.init(device.device_id_opencl);
	context.activate();

	gpu::gpu_mem_32u input_gpu;
	input_gpu.resizeN(n);
	input_gpu.writeN(as.data(), n);
	gpu::gpu_mem_32u output_gpu;
	output_gpu.resizeN(1);
	unsigned int workGroupSize = WORKGROUP_SIZE;

	{
		ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_1");

		bool printLog = false;
		kernel.compile(printLog);

		timer t;
		for (int iter = 0; iter < benchmarkingIters; ++iter) {
			unsigned int sum = 0;
			output_gpu.writeN(&sum, 1);
			kernel.exec(gpu::WorkSize(workGroupSize, n), input_gpu, output_gpu, n);
			output_gpu.readN(&sum, 1);
			EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
			t.nextLap();
		}

		std::cout << "GPU atomic:    " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
		std::cout << "GPU atomic:    " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
	}

	{
		ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_2");

		bool printLog = false;
		kernel.compile(printLog);

		timer t;
		for (int iter = 0; iter < benchmarkingIters; ++iter) {
			unsigned int sum = 0;
			unsigned int workGroupSize = 128;
			output_gpu.writeN(&sum, 1);
            unsigned int workSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
			kernel.exec(gpu::WorkSize(workGroupSize, workSize), input_gpu, output_gpu, n);
			output_gpu.readN(&sum, 1);
			EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
			t.nextLap();
		}

		std::cout << "GPU cycle:     " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
		std::cout << "GPU cycle:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
	}

	{
		ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_3");

		bool printLog = false;
		kernel.compile(printLog);

		timer t;
		for (int iter = 0; iter < benchmarkingIters; ++iter) {
			unsigned int sum = 0;
			output_gpu.writeN(&sum, 1);
			unsigned int workSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
			kernel.exec(gpu::WorkSize(workGroupSize, workSize), input_gpu, output_gpu, n);
			output_gpu.readN(&sum, 1);
			EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
			t.nextLap();
		}

		std::cout << "GPU coalesced: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
		std::cout << "GPU coalesced: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
	}

	{
		ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_4");

		bool printLog = false;
		kernel.compile(printLog);

		timer t;
		for (int iter = 0; iter < benchmarkingIters; ++iter) {
			unsigned int sum = 0;
			output_gpu.writeN(&sum, 1);
			kernel.exec(gpu::WorkSize(workGroupSize, n), input_gpu, output_gpu, n);
			output_gpu.readN(&sum, 1);
			EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
			t.nextLap();
		}

		std::cout << "GPU local buf: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
		std::cout << "GPU local buf: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
	}

	{
		ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_5");

		bool printLog = false;
		kernel.compile(printLog);

		timer t;
		for (int iter = 0; iter < benchmarkingIters; ++iter) {
			unsigned int sum = 0;
			output_gpu.writeN(&sum, 1);
			kernel.exec(gpu::WorkSize(workGroupSize, n), input_gpu, output_gpu, n);
			output_gpu.readN(&sum, 1);
			EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
			t.nextLap();
		}

		std::cout << "GPU tree:      " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
		std::cout << "GPU tree:      " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
	}

	{
		ocl::Kernel kernel(sum_kernel, sum_kernel_length, "sum_6");

		bool printLog = false;
		kernel.compile(printLog);
		gpu::gpu_mem_32u in_buffer, out_buffer;
		in_buffer.resizeN(n);
		out_buffer.resizeN(n);

		timer t;
		for (int iter = 0; iter < benchmarkingIters; ++iter) {
			in_buffer.writeN(as.data(), n);
			unsigned int sum = 0;
			for (unsigned int workSize = n; workSize > 1; workSize = (workSize + workGroupSize - 1) / workGroupSize) {
				kernel.exec(gpu::WorkSize(workGroupSize, workSize), in_buffer, out_buffer, workSize);
				std::swap(in_buffer, out_buffer);
			}
			// after swap, the result is in in_buffer
			in_buffer.readN(&sum, 1);
			EXPECT_THE_SAME(reference_sum, sum, "GPU result should be consistent!");
			t.nextLap();
		}

		std::cout << "GPU recursive: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
		std::cout << "GPU recursive: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl;
	}
}
