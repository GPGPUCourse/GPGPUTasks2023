//
// Created by evgenii on 08.03.24.
//

#ifndef APLUSB_HELPERS_H
#define APLUSB_HELPERS_H
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
#include <cstdint>
namespace helpers {
    template<typename T>
    std::string to_string(T value) {
        std::ostringstream ss;
        ss << value;
        return ss.str();
    }

    void reportError(cl_int err, const std::string& filename, int line) {
        if (CL_SUCCESS == err)
            return;

        // Таблица с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        std::string message =
                "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
        throw std::runtime_error(message);
    }

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

#define BUILD_PROGRAM_FAILURE -11

    std::vector<cl_platform_id> getPlatforms() {
        cl_uint platformsCount = 0;
        OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
        std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;
        std::vector<cl_platform_id> platforms(platformsCount);
        OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
        return platforms;
    }

    void printDeviceInfo(const cl_device_id& device) {
        std::size_t deviceNameSize;
        cl_device_type deviceType;

        OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
        std::vector<unsigned char> deviceName(deviceNameSize, 0);
        OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
        std::cout << "        Device name: " << deviceName.data() << std::endl;
        std::cout << "        Device Id: " << device << std::endl;

        OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
        std::cout << "        Device type: ";
        if (deviceType & CL_DEVICE_TYPE_CPU) {
            std::cout << "CPU ";
        }
        if (deviceType & CL_DEVICE_TYPE_GPU) {
            std::cout << "GPU ";
        }
        if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) {
            std::cout << "accelerator ";
        }
        if (deviceType & CL_DEVICE_TYPE_DEFAULT) {
            std::cout << "default ";
        }
        std::cout << std::endl;
    }

    std::vector<cl_device_id> getDevices(const cl_platform_id& platform) {
        cl_uint devicesCount = 0;
        const auto errcodeRet = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount);
        if (errcodeRet == CL_DEVICE_NOT_FOUND) {
            return {};
        }
        OCL_SAFE_CALL(errcodeRet);
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
        return devices;
    }

    std::vector<cl_device_id> getDevices(const std::vector<cl_platform_id>& platforms) {
        std::vector<cl_device_id> devices;
        for (const auto &platform : platforms) {
            auto platformDevices = getDevices(platform);
            devices.insert(devices.end(), platformDevices.begin(), platformDevices.end());
        }
        return devices;
    }

    cl_device_type getDeviceType(const cl_device_id& device) {
        cl_device_type deviceType;
        OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
        return deviceType;
    }

    cl_context createContext(const cl_device_id& device) {
        cl_int errcodeRet;
        auto context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &errcodeRet);
        OCL_SAFE_CALL(errcodeRet);
        return context;
    }

    cl_command_queue createQueue(const cl_context& context, const cl_device_id& device) {
        cl_int errcodeRet;
        auto queue = clCreateCommandQueue(context, device, 0, &errcodeRet);
        OCL_SAFE_CALL(errcodeRet);
        return queue;
    }

    cl_mem createBuffer(const cl_context& context, std::vector<float>&source,
                        const cl_mem_flags& memFlags) {
        cl_int errcodeRet;
        auto size = source.size() * sizeof(float);
        auto buffer = clCreateBuffer(context, memFlags | CL_MEM_USE_HOST_PTR, size, source.data(), &errcodeRet);
        OCL_SAFE_CALL(errcodeRet);
        return buffer;
    }

    cl_program createProgramWithSource(const cl_context &context, const std::string& source) {
        cl_int errcodeRet;
        size_t source_size = source.size();
        auto cString = source.c_str();
        auto program = clCreateProgramWithSource(context, 1, &cString, &source_size, &errcodeRet);
        OCL_SAFE_CALL(errcodeRet);
        return program;
    }

    void printBuildLog(const cl_program &program, const cl_device_id& device) {
        size_t logSize = 0;
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize));
        std::string log(logSize, '\n');
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr));
        std::cout << "Build log:"  << std::endl << "log size: " << logSize << std::endl << log << std::endl;
    }

    void buildProgram(const cl_program &program, const cl_device_id& device) {
        cl_int errcodeRet = (clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr));
        if (errcodeRet == BUILD_PROGRAM_FAILURE) {
            printBuildLog(program, device);
        }
        OCL_SAFE_CALL(errcodeRet);
    }

    std::vector<cl_kernel> getProgramKernels(const cl_program& program) {
        cl_uint size = 0;
        OCL_SAFE_CALL(clCreateKernelsInProgram(program, 0, nullptr, &size));
        std::vector<cl_kernel> kernels(size);
        OCL_SAFE_CALL(clCreateKernelsInProgram(program, size, kernels.data(), nullptr));
        if (kernels.empty()) {
            std::cerr << "No kernel found" << std::endl;
        }
        return kernels;
    }

    cl_uint getKernelArgsAmount(const cl_kernel& kernel) {
        cl_uint argsAmount = 0;
        OCL_SAFE_CALL(clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(argsAmount), &argsAmount, nullptr));
        return argsAmount;
    }

    cl_event enqueueLinearKernelExecution(
            const cl_command_queue& queue, const cl_kernel& kernel,
            const size_t&globalSize, const size_t& localSize) {
        cl_event event = nullptr;
        OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &globalSize, &localSize, 0, nullptr, &event));
        return event;
    }

    void waitForEvent(const cl_event& event) {
        OCL_SAFE_CALL(clWaitForEvents(1, &event));
    }

    template<class T>
    void setKernelArg(const cl_kernel& kernel, const unsigned int& argInd, const T& arg) {
        OCL_SAFE_CALL(clSetKernelArg(kernel, argInd, sizeof(T), &arg));
    }

    template<class T>
    void enqueRead(const cl_command_queue& queue,
                   const cl_mem& source, std::vector<T>& dest) {
        OCL_SAFE_CALL(clEnqueueReadBuffer(queue, source, CL_TRUE, 0, dest.size() * sizeof(T), dest.data(), 0, nullptr, nullptr));
    }

    void releaseKernel(cl_kernel& kernel) {
        OCL_SAFE_CALL(clReleaseKernel(kernel));
    }

    void releaseProgram(cl_program& program) {
        OCL_SAFE_CALL(clReleaseProgram(program));
    }

    void releaseMemObject(cl_mem& memObject) {
        OCL_SAFE_CALL(clReleaseMemObject(memObject));
    }

    void releaseQueue(cl_command_queue& queue) {
        OCL_SAFE_CALL(clReleaseCommandQueue(queue));
    }

    void releaseContext(cl_context& context) {
        OCL_SAFE_CALL(clReleaseContext(context));
    }
} // namespace helpers
#endif//APLUSB_HELPERS_H
