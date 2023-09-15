#pragma once
#include <CL/cl.h>
#include <algorithm>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cl_error_print.h"


#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

template<class ParamT, class ObjT, class OcvFuncT>
ParamT getInfo(ObjT object, cl_device_info param, OcvFuncT clFunc) {
    ParamT paramValue;
    OCL_SAFE_CALL(clFunc(object, param, sizeof(ParamT), &paramValue, nullptr));
    return paramValue;
}

template<class ObjT, class OcvFuncT>
std::string getStringInfo(ObjT object, cl_device_info param, OcvFuncT ocvFunc) {
    size_t size = 0;
    OCL_SAFE_CALL(ocvFunc(object, param, 0, nullptr, &size));
    auto paramValue = std::string(size, 0);
    OCL_SAFE_CALL(ocvFunc(object, param, size, &paramValue[0], nullptr));
    return paramValue;
}

inline std::vector<cl_platform_id> getPlatforms() {
    cl_uint cnt = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &cnt));
    std::vector<cl_platform_id> platforms(cnt);
    OCL_SAFE_CALL(clGetPlatformIDs(cnt, platforms.data(), nullptr));
    return platforms;
}

inline std::vector<cl_device_id> getDevices(cl_platform_id platform, cl_device_type deviceType = CL_DEVICE_TYPE_ALL) {
    cl_uint devicesCount = 0;
    auto retCode = clGetDeviceIDs(platform, deviceType, 0, nullptr, &devicesCount);
    if (retCode == CL_DEVICE_NOT_FOUND)
        return {};
    OCL_SAFE_CALL(retCode);
    std::vector<cl_device_id> devices(devicesCount);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, deviceType, devicesCount, devices.data(), nullptr));
    return devices;
}


inline std::string getDeviceStringInfo(cl_device_id device, cl_device_info param) {
    return getStringInfo(device, param, clGetDeviceInfo);
}

template<class T>
T getDeviceInfo(cl_device_id device, cl_device_info param) {
    return getInfo<T>(device, param, clGetDeviceInfo);
}


inline cl_uint getDeviceCU(cl_device_id device) {
    return getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS);
}

inline std::string getDeviceName(cl_device_id device) {
    return getDeviceStringInfo(device, CL_DEVICE_NAME);
}


inline bool hasDeviceType(cl_device_id device, cl_device_type type) {
    return getDeviceInfo<cl_device_type>(device, CL_DEVICE_TYPE) & type;
}


inline bool isGPUDevice(cl_device_id device) {
    return hasDeviceType(device, CL_DEVICE_TYPE_GPU);
}

inline std::vector<cl_device_id> getDevices(cl_device_type type) {
    auto platforms = getPlatforms();
    std::vector<cl_device_id> gpuDevices;
    for (auto &platform : platforms) {
        auto curDevices = getDevices(platform, type);
        gpuDevices.insert(gpuDevices.end(), curDevices.begin(), curDevices.end());
    }
    return gpuDevices;
}


inline cl_device_id getDeviceWithMaxCU(const std::vector<cl_device_id> &devices) {
    return *std::max_element(devices.begin(), devices.end(),
                             [](cl_device_id lhs, cl_device_id rhs) { return getDeviceCU(lhs) < getDeviceCU(rhs); });
}

inline cl_device_id getDeviceWithMinCU(const std::vector<cl_device_id> &devices) {
    return *std::max_element(devices.begin(), devices.end(),
                             [](cl_device_id lhs, cl_device_id rhs) { return getDeviceCU(lhs) > getDeviceCU(rhs); });
}


inline cl_context createContext(cl_device_id device) {
    cl_int retCode;
    auto context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &retCode);
    OCL_SAFE_CALL(retCode);
    return context;
}

inline void releaseContext(cl_context context) {
    OCL_SAFE_CALL(clReleaseContext(context));
}


inline cl_command_queue createInorderQueue(cl_context context, cl_device_id device) {
    cl_int retCode;
    auto queue = clCreateCommandQueue(context, device, cl_command_queue_properties{0}, &retCode);
    OCL_SAFE_CALL(retCode);
    return queue;
}


inline void releaseQueue(cl_command_queue queue) {
    OCL_SAFE_CALL(clReleaseCommandQueue(queue));
}

template<class T>
void writeToBuffer(cl_command_queue queue, cl_mem buffer, std::vector<T> values) {
    auto size = values.size() * sizeof(T);
    OCL_SAFE_CALL(clEnqueueWriteBuffer(queue, buffer, true, 0, size, values.data(), 0, nullptr, nullptr));
}


template<class T>
cl_mem createBufferFrom(cl_context context, std::vector<T> &from, cl_mem_flags flags = {}) {
    cl_int retCode;
    auto size = from.size() * sizeof(T);
    auto buffer = clCreateBuffer(context, flags | CL_MEM_COPY_HOST_PTR, size, from.data(), &retCode);
    OCL_SAFE_CALL(retCode);
    return buffer;
}


template<class T>
cl_mem createBufferWithSize(cl_context context, size_t count, cl_mem_flags flags = {}) {
    cl_int retCode;
    auto size = count * sizeof(T);
    auto buffer = clCreateBuffer(context, flags, size, nullptr, &retCode);
    OCL_SAFE_CALL(retCode);
    return buffer;
}

inline void releaseBuffer(cl_mem buffer) {
    OCL_SAFE_CALL(clReleaseMemObject(buffer));
}

inline cl_program createProgramWithSource(cl_context context, const std::string &source) {
    cl_int retCode;
    auto *source_ptr = &source[0];
    size_t source_size = source.size();
    auto program = clCreateProgramWithSource(context, 1, &source_ptr, &source_size, &retCode);
    OCL_SAFE_CALL(retCode);
    return program;
}

inline void releaseProgram(cl_program program) {
    OCL_SAFE_CALL(clReleaseProgram(program));
}

inline cl_int buildProgram(cl_program program, cl_device_id device) {
    return clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
}

inline std::string getBuildLog(cl_program program, cl_device_id device) {
    size_t logSize{};
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize));
    std::string log(logSize, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, logSize, &log[0], nullptr));
    return log;
}


inline std::vector<cl_kernel> createKernelsInProgram(cl_program program) {
    cl_uint size{};
    OCL_SAFE_CALL(clCreateKernelsInProgram(program, 0, nullptr, &size));
    std::vector<cl_kernel> kernels(size);
    OCL_SAFE_CALL(clCreateKernelsInProgram(program, size, kernels.data(), nullptr));
    return kernels;
}

inline void releaseKernel(cl_kernel kernel) {
    OCL_SAFE_CALL(clReleaseKernel(kernel));
}

inline cl_event enqueueOneDimKernelExecution(cl_command_queue queue, cl_kernel kernel, size_t global_size,
                                             size_t local_size) {
    cl_event event{};
    OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_size, &local_size, 0, nullptr, &event));
    return event;
}

inline void waitForEvent(cl_event event) {
    OCL_SAFE_CALL(clWaitForEvents(1, &event));
}

template<class T>
inline void setKernelArg(cl_kernel kernel, unsigned int argIndex, const T &arg) {
    OCL_SAFE_CALL(clSetKernelArg(kernel, argIndex, sizeof(T), &arg));
}


template<class T>
inline std::vector<T> readBuffer(cl_command_queue queue, cl_mem buffer, size_t size) {
    std::vector<T> values(size);
    readBufferTo(queue, buffer, size, values);
    return values;
}

template<class T>
inline void readBufferTo(cl_command_queue queue, cl_mem buffer, size_t size, std::vector<T>& to) {
    OCL_SAFE_CALL(clEnqueueReadBuffer(queue, buffer, CL_TRUE, 0, size * sizeof(T), to.data(), 0, nullptr, nullptr));
}