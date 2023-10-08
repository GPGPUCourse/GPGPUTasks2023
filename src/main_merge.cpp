#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/merge_cl.h"

#include <iostream>
#include <stdexcept>
#include <vector>


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
    unsigned int n = 32 * 1024 * 128;
    std::vector<float> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<float> cpu_sorted;
    {
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            cpu_sorted = as;
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
    }

    gpu::gpu_mem_32f as_gpu;
    gpu::gpu_mem_32f ss_gpu;
    as_gpu.resizeN(n);
    ss_gpu.resizeN(n);
    {
        const unsigned int work_per_workitem = 32;

        ocl::Kernel merge1(
            merge_kernel, 
            merge_kernel_length, 
            "merge_two_iters",
            "-DWORK_PER_WORKITEM=" + std::to_string(work_per_workitem) 
        );
        merge1.compile();

        ocl::Kernel merge2(
            merge_kernel, 
            merge_kernel_length, 
            "merge_local",
            "-DWORK_PER_WORKITEM=" + std::to_string(work_per_workitem)   
        );
        merge2.compile();

        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            as_gpu.writeN(as.data(), n);
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных
            const unsigned int workGroupSize = 64;
            const unsigned int warpsCount = 32; // сколько одновременно рабочих групп мы можем исполнять
            
            // пока есть возможность мерджить по два массива без простоя варпов - делаем так
            // *(workGroupSize*warpsCount)
            int h;
            for(h=1;(1<<h)<=n/(workGroupSize*warpsCount);++h) {
                unsigned int global_work_size = n / (1<<h);
                //global_work_size = std::max(global_work_size, workGroupSize);
                merge1.exec(
                    gpu::WorkSize(workGroupSize, global_work_size), 
                    as_gpu, 
                    ss_gpu, 
                    n, 
                    (1<<h)
                );
                std::swap(as_gpu, ss_gpu);
            }

            // теперь будем мерджить два (уже больших) массива всеми воркитемами (каждый посчитает свой подотрезок ответа)
            for(;(1<<h)<=n;++h) {
                unsigned int len = (1<<h);
                unsigned int global_work_size = len / work_per_workitem;
                for(int j=0;j<n;j+=len) {
                    merge2.exec(
                        gpu::WorkSize(workGroupSize, global_work_size), 
                        as_gpu, 
                        ss_gpu, 
                        n,
                        len,
                        j
                    );
                }
                std::swap(as_gpu, ss_gpu);
            }
            
            t.nextLap();
        }
        std::cout << "GPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU: " << (n / 1000 / 1000) / t.lapAvg() << " millions/s" << std::endl;
        as_gpu.readN(as.data(), n);
    }
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
