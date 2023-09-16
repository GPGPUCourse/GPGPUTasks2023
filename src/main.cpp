#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <tuple>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

std::vector<cl_platform_id> findPlatforms() {
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
    return platforms;
}

std::vector<cl_device_id> findDevices(cl_platform_id platform) {
    cl_uint devicesCount = 0;
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
    std::vector<cl_device_id> devices(devicesCount);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
    return devices;
}

cl_device_type getDeviceType(cl_device_id device) {
    cl_device_type deviceType;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
    return deviceType;
}

std::pair<cl_device_id, cl_platform_id> selectDevice() {
    std::vector<std::pair<cl_device_id, cl_platform_id>> CPUs;
    auto platforms = findPlatforms();
    for (auto platform: platforms) {
        auto devices = findDevices(platform);
        for (auto device: devices) {
            auto deviceType = getDeviceType(device);
            if (deviceType == CL_DEVICE_TYPE_GPU)
                return {device, platform};
            else if (deviceType == CL_DEVICE_TYPE_CPU)
                CPUs.emplace_back(device, platform);
        }
    }
    if (CPUs.empty())
        throw std::runtime_error("Cannot find any appropriate device");
    return CPUs.front();
}

cl_context createContext(cl_device_id device, cl_platform_id platform) {
    cl_context_properties contextProperties[] = {CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0};
    cl_int err;
    cl_context context = clCreateContext(contextProperties, 1, &device, nullptr, nullptr, &err);
    OCL_SAFE_CALL(err);
    return context;
}

cl_command_queue createCommandQueue(cl_context context, cl_device_id device) {
    cl_int err;
    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    OCL_SAFE_CALL(err);
    return queue;
}

template<typename T>
cl_mem createBuffer(cl_context context, cl_mem_flags flags, std::vector<T> &vector) {
    cl_int err;
    cl_mem buf = clCreateBuffer(context, flags, sizeof(T) * vector.size(), vector.data(), &err);
    OCL_SAFE_CALL(err);
    return buf;
}

cl_program createProgramWithSource(cl_context context, const std::string &filePath) {
    std::string kernel_sources;
    {
        std::ifstream file(filePath);
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
//         std::cout << kernel_sources << std::endl;
    }

    const char *sources[] = {kernel_sources.c_str()};
    size_t lengths[] = {kernel_sources.size()};
    cl_int err;
    cl_program program = clCreateProgramWithSource(context, 1, sources, lengths, &err);
    OCL_SAFE_CALL(err);
    return program;
}

void compileProgram(cl_program program, cl_device_id device) {
    cl_int err = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));

    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));

    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }
    OCL_SAFE_CALL(err);
}


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    cl_device_id device;
    cl_platform_id platform;
    std::tie(device, platform) = selectDevice();

    auto context = createContext(device, platform);
    auto commandQueue = createCommandQueue(context, device);

    unsigned int n = 100 * 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    auto aBuf = createBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, as);
    auto bBuf = createBuffer(context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, bs);
    auto cBuf = createBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR, cs);

    cl_program aplusbProgram = createProgramWithSource(context, "src/cl/aplusb.cl");

    compileProgram(aplusbProgram, device);

    cl_int err;
    cl_kernel aplusbKernel = clCreateKernel(aplusbProgram, "aplusb", &err);
    OCL_SAFE_CALL(err);


    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(aplusbKernel, i++, sizeof(cl_mem), &aBuf));
        OCL_SAFE_CALL(clSetKernelArg(aplusbKernel, i++, sizeof(cl_mem), &bBuf));
        OCL_SAFE_CALL(clSetKernelArg(aplusbKernel, i++, sizeof(cl_mem), &cBuf));
        OCL_SAFE_CALL(clSetKernelArg(aplusbKernel, i++, sizeof(unsigned int), &n));
    }

    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;// Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        cl_event event;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(
                    clEnqueueNDRangeKernel(commandQueue, aplusbKernel, 1, nullptr, &global_work_size, &workGroupSize, 0,
                                           nullptr, &event));
            clWaitForEvents(1, &event);
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        std::cout << "GFlops: " << n / t.lapAvg() / 1e9 << std::endl;

        std::cout << "VRAM bandwidth: " << double(n * sizeof(float) * 3) / t.lapAvg() / (1024 * 1024 * 1024) << " GB/s"
                  << std::endl;
        OCL_SAFE_CALL(clReleaseEvent(event));
    }

    {
        cl_event event;
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(commandQueue, cBuf, CL_TRUE, 0, sizeof(float) * n, cs.data(), 0, nullptr,
                                              &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            t.nextLap();
        }
        OCL_SAFE_CALL(clReleaseEvent(event));
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << double(n * sizeof(float)) / t.lapAvg() / (1 << 30) << " GB/s" << std::endl;
    }

    // Проверил на kernel, где c[i] = a[i] + b[i] + 1;
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    OCL_SAFE_CALL(clReleaseKernel(aplusbKernel));
    OCL_SAFE_CALL(clReleaseProgram(aplusbProgram));
    OCL_SAFE_CALL(clReleaseMemObject(aBuf));
    OCL_SAFE_CALL(clReleaseMemObject(bBuf));
    OCL_SAFE_CALL(clReleaseMemObject(cBuf));
    OCL_SAFE_CALL(clReleaseCommandQueue(commandQueue));
    OCL_SAFE_CALL(clReleaseContext(context));
    return 0;
}
