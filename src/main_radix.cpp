#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

// Этот файл будет сгенерирован автоматически в момент сборки - см. convertIntoHeader в CMakeLists.txt:18
#include "cl/radix_cl.h"
#include "cl/matrix_transpose_cl.h"
#include "cl/merge_cl.h"
#include "cl/prefix_sum_cl.h"

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

ocl::Kernel get_kernel(const char *file, size_t file_len, std::string kernel_name, std::string defines = "") {
    ocl::Kernel kernel(file, file_len, kernel_name, defines);
    kernel.compile();
    return kernel;
}

void print_gpu_mem(gpu::gpu_mem_32u as_gpu, unsigned int n) {
    std::vector<unsigned int> res(n, 0);
    as_gpu.readN(res.data(), n);

    for (auto a : res)
        std::cout << a << ' ';
    std::cout << '\n';
}

void print_gpu_mem(gpu::gpu_mem_32u as_gpu, unsigned int n, unsigned int m) {
    std::vector<unsigned int> res(n * m, 0);
    as_gpu.readN(res.data(), n * m);

    std::cout<<"n and m "<<n<<" "<<m<<"\n";
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            std::cout << res[i * m + j] << ' ';
        std::cout << '\n';
    }
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    unsigned int n = 32 * 1024 * 1024;
    std::vector<unsigned int> as(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = (unsigned int) r.next(0, std::numeric_limits<int>::max());
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    std::vector<unsigned int> cpu_sorted;
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

    {
        gpu::gpu_mem_32u as_gpu, temp_gpu, cs_gpu, cs_t_gpu, pref_gpu, pref_t_gpu;
        as_gpu.resizeN(n);
        temp_gpu.resizeN(n);
        cs_gpu.resizeN(n);
        cs_t_gpu.resizeN(n);
        pref_gpu.resizeN(n);
        pref_t_gpu.resizeN(n);

        {
            const unsigned int k = 4;
            const unsigned int WGT = 16;
            const unsigned int WG = 128;

            ocl::Kernel merge_sort_kernel = get_kernel(
                merge_kernel, 
                merge_kernel_length, 
                "merge"
            );

            ocl::Kernel transpose_kernel = get_kernel(
                matrix_transpose_kernel, 
                matrix_transpose_kernel_length, 
                "matrix_transpose", 
                ("-DWG0=" + std::to_string(WGT) + " -DWG1=" + std::to_string(WGT))
            );

            ocl::Kernel prefix_sum = get_kernel(
                prefix_sum_kernel, 
                prefix_sum_kernel_length, 
                "prefix_sum", 
                "-DWG=" + to_string(WG) + " -DSZ=" + to_string(WGT)
            );

            ocl::Kernel block_prefix_sum = get_kernel(
                prefix_sum_kernel, 
                prefix_sum_kernel_length, 
                "block_prefix_sum", 
                "-DWG=" + to_string(WG) + " -DSZ=" + to_string(WGT)
            );

            ocl::Kernel accum = get_kernel(
                prefix_sum_kernel, 
                prefix_sum_kernel_length, 
                "accum", 
                "-DWG=" + to_string(WG) + " -DSZ=" + to_string(WGT)
            );

            ocl::Kernel count = get_kernel(
                radix_kernel, 
                radix_kernel_length, 
                "count", 
                "-DWG=" + to_string(WG) + " -DSZ=" + to_string(WGT)
            );

            ocl::Kernel radix = get_kernel(
                radix_kernel, 
                radix_kernel_length, 
                "radix",
                "-DWG=" + to_string(WG) + " -DSZ=" + to_string(1<<k)
            );

            ocl::Kernel fill = get_kernel(
                radix_kernel, 
                radix_kernel_length, 
                "fill",
                "-DWG=" + to_string(WG) + " -DSZ=" + to_string(1<<k)
            );

            timer t;
            for (int iter = 0; iter < benchmarkingIters; ++iter) {
                as_gpu.writeN(as.data(), n);

                // Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфер данных
                t.restart();

                const unsigned int work_group_size = WG;
                unsigned int work_group_ct = n / work_group_size;
                for (int shift = 0; shift < sizeof(unsigned int) * 8; shift += k)
                {
                    // temp_gpu.writeN(n_zeros.data(), n);
                    // cs_gpu.writeN(n_zeros.data(), n);
                    // cs_t_gpu.writeN(n_zeros.data(), n);
                    // pref_gpu.writeN(n_zeros.data(), n);
                    // pref_t_gpu.writeN(n_zeros.data(), n);

                    fill.exec(gpu::WorkSize(work_group_size, n), temp_gpu, 0, n);
                    fill.exec(gpu::WorkSize(work_group_size, n), cs_gpu, 0, n);
                    fill.exec(gpu::WorkSize(work_group_size, n), cs_t_gpu, 0, n);
                    fill.exec(gpu::WorkSize(work_group_size, n), pref_gpu, 0, n);
                    fill.exec(gpu::WorkSize(work_group_size, n), pref_t_gpu, 0, n);

                    //std::cout << "as_gpu\n";
                    //print_gpu_mem(as_gpu, n);

                    // merge sort 
                    {
                        {
                            unsigned int cur = 1;
                            while (cur < work_group_size) {
                                merge_sort_kernel.exec(
                                    gpu::WorkSize(std::min(n, 128u), n), 
                                    as_gpu, 
                                    temp_gpu, 
                                    n,
                                    cur, 
                                    shift, 
                                    k
                                );
                                cur <<= 1;
                                std::swap(as_gpu, temp_gpu);
                            }
                        }
                    }   
                    // std::cout << "as_gpu merge\n";
                    // print_gpu_mem(as_gpu, n);

                    // calc counter 
                    {
                        //cs_gpu.writeN(n_zeros.data(), n);
                        fill.exec(gpu::WorkSize(work_group_size, n), cs_gpu, 0, n);
                        count.exec(
                            gpu::WorkSize(work_group_size, n), 
                            as_gpu, 
                            cs_gpu, 
                            shift, 
                            k
                        );
                    }
                    // std::cout << "cs_gpu\n";
                    // print_gpu_mem(cs_gpu, work_group_ct, WGT);

                    // transpose
                    {   
                        // any (x,y) such that x is a divisor of (1<<k) + y is a divisor of n/(work_group_size=128)
                        transpose_kernel.exec(
                            gpu::WorkSize(WGT, WGT, WGT, work_group_ct), 
                            cs_gpu, 
                            cs_t_gpu, 
                            WGT, 
                            work_group_ct
                        );
                    }
                    // std::cout << "cs_t_gpu\n";
                    // print_gpu_mem(cs_t_gpu, WGT, work_group_ct);
    
                    // prefix sums 
                    {
                        unsigned int work_group_size = std::min(n/WGT, 128u);

                        block_prefix_sum.exec(
                            gpu::WorkSize(work_group_size, n/WGT), 
                            pref_gpu, 
                            cs_gpu
                        );
                    }
                    // std::cout << "pref_gpu\n";
                    // print_gpu_mem(pref_gpu, work_group_ct, WGT);
                    
                    // prefix sums T
                    {
                        unsigned int work_group_size = std::min(n, 128u);

                        prefix_sum.exec(
                            gpu::WorkSize(work_group_size, n), 
                            temp_gpu, 
                            cs_t_gpu,
                            0
                        );
                        prefix_sum.exec(
                            gpu::WorkSize(work_group_size, n), 
                            pref_t_gpu, 
                            temp_gpu,
                            1
                        );
                        for(int i=2;i<=n;i<<=1) {
                            accum.exec(
                                gpu::WorkSize(work_group_size, ((n/i)+work_group_size-1)/work_group_size*work_group_size), 
                                temp_gpu, 
                                i,
                                n
                            );
                            prefix_sum.exec(
                                gpu::WorkSize(work_group_size, n), 
                                pref_t_gpu, 
                                temp_gpu,
                                i
                            );
                        }
                    }
                    // std::cout << "pref_t_gpu\n";
                    // print_gpu_mem(pref_t_gpu, WGT, work_group_ct);

                    // radix
                    {
                        radix.exec(
                            gpu::WorkSize(work_group_size, n), 
                            as_gpu, 
                            cs_gpu,
                            cs_t_gpu,
                            pref_gpu, 
                            pref_t_gpu, 
                            temp_gpu, 
                            shift, 
                            k
                        );
                    }
                    // std::cout << "radix\n";
                    // print_gpu_mem(temp_gpu, n);

                    std::swap(as_gpu, temp_gpu);
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
    }

    return 0;
}
