#include "cl/sum_cl.h"
#include "libgpu/context.h"
#include "libgpu/shared_device_buffer.h"
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>
#define WORKGROUP_SIZE 64
#define VALUES_PER_WORKITEM 64

const int benchmarkingIters = 10;
unsigned int reference_sum = 0;
const unsigned int n = 100*1000*1000;
const unsigned int zero[] = {0};
char defines[1000];


template<typename T>
void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line)
{
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

void measure_time_gpu(gpu::gpu_mem_32u as_gpu, std::string method_name, unsigned int size)
{

    gpu::gpu_mem_32u result_gpu;
    result_gpu.resizeN(1);

    ocl::Kernel kernel(sum_kernel,
                       sum_kernel_length,
                       method_name,
                       defines);

    bool printLog = false;
    kernel.compile(printLog);

    unsigned int workSizeX = (size + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;
    auto workSize = gpu::WorkSize(WORKGROUP_SIZE, workSizeX);

    timer t;
    for (int iter = 0; iter < benchmarkingIters; ++iter) {
        result_gpu.writeN(zero, 1);
        kernel.exec(workSize,as_gpu, result_gpu, n);

        unsigned int sum = 0;
        result_gpu.readN(&sum, 1);

        EXPECT_THE_SAME(reference_sum, sum, "GPU " + method_name + " result should be consistent!");
        t.nextLap();
    }
    std::cout << "GPU " + method_name + ": " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
    std::cout << "GPU " + method_name + ": " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl << std::endl;
}


int main(int argc, char **argv)
{
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
        std::cout << "CPU:     " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl << std::endl;
    }
            unsigned int global_work_size = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE * WORKGROUP_SIZE;
            timer t;

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
        std::cout << "CPU OMP: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU OMP: " << (n/1000.0/1000.0) / t.lapAvg() << " millions/s" << std::endl << std::endl;
    }

    {
        // TODO: implement on OpenCL
         gpu::Device device = gpu::chooseGPUDevice(argc, argv);
         gpu::Context context;
         context.init(device.device_id_opencl);
         context.activate();

         gpu::gpu_mem_32u as_gpu;
         as_gpu.resizeN(n);
         as_gpu.writeN(as.data(), n);

         sprintf(defines, "-D VALUES_PER_WORKITEM=%d -D WORKGROUP_SIZE=%d", VALUES_PER_WORKITEM, WORKGROUP_SIZE);


         measure_time_gpu(as_gpu, "sum1_baseline_atomic_add", n);
         auto temp = (n + VALUES_PER_WORKITEM - 1) / VALUES_PER_WORKITEM;
         measure_time_gpu(as_gpu, "sum2_with_cycle", temp);
         measure_time_gpu(as_gpu, "sum3_with_cycle_coalesced", temp);
         measure_time_gpu(as_gpu, "sum4_local_mem_and_main_thread", n);
         measure_time_gpu(as_gpu, "sum5_with_tree", n);

    }
}
