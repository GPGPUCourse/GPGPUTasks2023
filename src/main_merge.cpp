#include <libgpu/context.h>
#include <libgpu/shared_device_buffer.h>
#include <libutils/fast_random.h>
#include <libutils/misc.h>
#include <libutils/timer.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>


template<typename T>
void raiseFail(const T &a, const T &b, const char *message, const char *filename, int line) {
    if (a != b) {
        std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
        throw std::runtime_error(message);
    }
}

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

std::string read_kernel(const std::string &path) {
    std::ifstream fin(path);
    std::ostringstream oss;
    oss << fin.rdbuf();
    return oss.str();
}

int main(int argc, char **argv) {
    gpu::Device device = gpu::chooseGPUDevice(argc, argv);

    gpu::Context context;
    context.init(device.device_id_opencl);
    context.activate();

    int benchmarkingIters = 10;
    constexpr cl_uint n = 32 * 1024 * 1024;
    constexpr cl_uint workGroupSize = 64;
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
            //for (int i = 0; i < n; i += workGroupSize) {
            //    std::sort(cpu_sorted.begin() + i, cpu_sorted.begin() + i + workGroupSize);
            //}
            std::sort(cpu_sorted.begin(), cpu_sorted.end());
            t.nextLap();
        }
        std::cout << "CPU: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "CPU: " << 1e-6 * n / t.lapAvg() << " millions/s" << std::endl;
    }

    std::string base_path = __FILE__;
    base_path = base_path.substr(0, base_path.size() - 14);

    gpu::WorkSize ws(workGroupSize, n);
    std::ostringstream defines;
    defines << "-DWORK_GROUP_SIZE=" << workGroupSize;

    gpu::gpu_mem_32f src_gpu;
    gpu::gpu_mem_32f dst_gpu;
    gpu::gpu_mem_32u idx_gpu;
    src_gpu.resizeN(n);
    dst_gpu.resizeN(n);

    {
        std::string src = read_kernel(base_path + "merge_simple.cl");
        ocl::Kernel merge(src.c_str(), src.size(), "kmain", defines.str());
        merge.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            src_gpu.writeN(as.data(), n);
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных

            for (cl_uint sortedSize = 1; sortedSize < n; sortedSize *= 2) {
                merge.exec(ws, src_gpu, dst_gpu, n, sortedSize);
                std::swap(src_gpu, dst_gpu);
            }

            t.nextLap();
        }
        std::cout << "GPU one phase: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU one phase: " << 1e-6 * n / t.lapAvg() << " millions/s" << std::endl;
        src_gpu.readN(as.data(), n);
    }

    {
        std::string src1 = read_kernel(base_path + "merge_simple_local.cl");
        std::string src2 = read_kernel(base_path + "merge_simple.cl");
        ocl::Kernel phase1(src1.c_str(), src1.size(), "kmain", defines.str());
        ocl::Kernel phase2(src2.c_str(), src2.size(), "kmain", defines.str());
        phase1.compile();
        phase2.compile();
        timer t;
        for (int iter = 0; iter < benchmarkingIters; ++iter) {
            src_gpu.writeN(as.data(), n);
            t.restart();// Запускаем секундомер после прогрузки данных, чтобы замерять время работы кернела, а не трансфера данных

            // Выделение фазы, когда сортируемый блок попадает в локальную память,
            // и можно использовать синхронизацию барьерами, увеличивает производительность
            // с 193 млн/с до 356 млн/с на GTX 1060, т.е. почти в два раза.
            phase1.exec(ws, src_gpu, dst_gpu, n);
            std::swap(src_gpu, dst_gpu);

            for (cl_uint sortedSize = workGroupSize; sortedSize < n; sortedSize *= 2) {
                phase2.exec(ws, src_gpu, dst_gpu, n, sortedSize);
                std::swap(src_gpu, dst_gpu);
            }

            t.nextLap();
        }
        std::cout << "GPU two phases: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "GPU two phases: " << 1e-6 * n / t.lapAvg() << " millions/s" << std::endl;
        src_gpu.readN(as.data(), n);
    }

    //for (int i = 0; i < n; ++i) {
    //    std::cerr << "i=" << i << "\tcpu=" << cpu_sorted[i] << "\tgpu=" << as[i] << "\n";
    //}
    // Проверяем корректность результатов
    for (int i = 0; i < n; ++i) {
        EXPECT_THE_SAME(as[i], cpu_sorted[i], "GPU results should be equal to CPU results!");
    }

    return 0;
}
