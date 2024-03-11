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

void showPlatformInfo(cl_platform_id platform) {
    size_t platformNameSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
    std::vector<unsigned char> platformName(platformNameSize, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
    std::cout << "    Platform name: " << platformName.data() << std::endl;

    size_t vendorNameSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &vendorNameSize));
    std::vector<unsigned char> vendorName(vendorNameSize, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, vendorNameSize, vendorName.data(), nullptr));
    std::cout << "    Vendor: " << vendorName.data() << std::endl;
}

cl_device_type showDeviceInfo(cl_device_id deviceId) {
    size_t deviceNameSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
    std::vector<unsigned char> deviceName(deviceNameSize, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
    std::cout << "        Device name: " << deviceName.data() << std::endl;

    cl_device_type deviceType;
    OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
    std::cout << "        Device type: ";
    if (deviceType == CL_DEVICE_TYPE_GPU)
        std::cout << "GPU ";
    else if (deviceType == CL_DEVICE_TYPE_CPU)
        std::cout   << "CPU ";
    else if (deviceType == CL_DEVICE_TYPE_ACCELERATOR)
        std::cout << "ACCELERATOR ";
    else if (deviceType == CL_DEVICE_TYPE_DEFAULT)
        std::cout << "DEFAULT TYPE ";
    std::cout << std::endl;

    return deviceType;
}

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    cl_device_id currentDevice = nullptr;

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        cl_platform_id platform = platforms[platformIndex];
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        showPlatformInfo(platform);

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            cl_device_id deviceId = devices[deviceIndex];
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;

            cl_device_type deviceType = showDeviceInfo(deviceId);
            if (deviceType == CL_DEVICE_TYPE_GPU) {
                currentDevice = deviceId;
                break;
            }

            if (currentDevice == nullptr && deviceType == CL_DEVICE_TYPE_CPU) {
                currentDevice = deviceId;
            }
        }
    }

    if (currentDevice == nullptr) {
        throw std::runtime_error("The GPU/CPU device could not be found");
    }

    cl_int error;
    cl_context context = clCreateContext(nullptr, 1, &currentDevice, nullptr, nullptr, &error);
    OCL_SAFE_CALL(error);

    cl_command_queue commandQueue = clCreateCommandQueue(context, currentDevice, 0, &error);
    OCL_SAFE_CALL(error);

    uint n = 100 * 1000 * 1000;
    std::vector<float> firstArray(n, 0);
    std::vector<float> secondArray(n, 0);
    std::vector<float> result(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        firstArray[i] = r.nextf();
        secondArray[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    size_t sizeBuffer = n * sizeof(float);
    cl_mem_flags flagUseHost = CL_MEM_USE_HOST_PTR;
    cl_mem firstArrayBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | flagUseHost, sizeBuffer, firstArray.data(), &error);
    OCL_SAFE_CALL(error);
    cl_mem secondArrayBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY | flagUseHost, sizeBuffer, secondArray.data(), &error);
    OCL_SAFE_CALL(error);
    cl_mem resultBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeBuffer, nullptr, &error);
    OCL_SAFE_CALL(error);

    std::string kernelSources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernelSources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernelSources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
    }

    const char* kernelSourcesInCStr = kernelSources.c_str();
    size_t sizeKernelSource = kernelSources.size();
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSourcesInCStr, &sizeKernelSource, &error);
    OCL_SAFE_CALL(error);

    error = clBuildProgram(program, 1, &currentDevice, nullptr, nullptr, nullptr);
    if (error == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size = 0;
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, currentDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
        std::vector<char> log(log_size, 0);
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, currentDevice, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
        if (log_size > 1) {
            std::cout << "Log:" << std::endl;
            std::cout << log.data() << std::endl;
        }
    }
    OCL_SAFE_CALL(error);
    cl_kernel kernel = clCreateKernel(program, "aplusb", &error);
    OCL_SAFE_CALL(error);

    {
        unsigned int i = 0;
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &firstArrayBuffer));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &secondArrayBuffer));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(cl_mem), &resultBuffer));
        OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(uint), &n));
    }

    {
        size_t workGroupSize = 128;
        size_t globalWorkSize = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer timer;
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(commandQueue, kernel, 1, nullptr, &globalWorkSize, &workGroupSize, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
            clReleaseEvent(event);
            timer.nextLap();
        }

        std::cout << "Kernel average time: " << timer.lapAvg() << "+-" << timer.lapStd() << " s" << std::endl;
        std::cout << "GFlops: " << n / 1e9 / timer.lapAvg() << std::endl;
        std::cout << "VRAM bandwidth: " << 3.0 * n * sizeof(float) / (1024 * 1024 * 1024) / timer.lapAvg() << " GB/s" << std::endl;
    }

    {
        timer timer;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(commandQueue, resultBuffer, CL_TRUE, 0, sizeBuffer, result.data(), 0, nullptr, nullptr));
            timer.nextLap();
        }
        std::cout << "Result data transfer time: " << timer.lapAvg() << "+-" << timer.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << (double)sizeBuffer / (1024 * 1024 * 1024) / timer.lapAvg() << " GB/s" << std::endl;
    }

    for (unsigned int i = 0; i < n; ++i) {
        if (result[i] != firstArray[i] + secondArray[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    return 0;
}