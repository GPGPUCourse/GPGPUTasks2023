#pragma once

#include <CL/cl.h>
#include <string>
#include "Util.h"

namespace cl_objects {
    class Device {
        cl_device_id id;
    public:
        Device() : id(nullptr) {}

        explicit Device(cl_device_id id) : id(id) {}

        Device(const Device &other) : id(other.id) {}

        const cl_device_id &getId() const {
            return id;
        }

        std::string type() const {
            auto type = getInfo<cl_device_type>(clGetDeviceInfo, id, CL_DEVICE_TYPE);
            if (type & CL_DEVICE_TYPE_GPU)
                return "GPU";
            else if (type * CL_DEVICE_TYPE_CPU)
                return "CPU";
        }

        std::string name() const {
            auto nameVec = getInfoVec<unsigned char, size_t>(clGetDeviceInfo, id, CL_DEVICE_NAME);
            return {nameVec.begin(), nameVec.end() - 1};
        }

        cl_ulong globalSize() const {
            return getInfo<cl_ulong>(clGetDeviceInfo, id, CL_DEVICE_GLOBAL_MEM_SIZE);
        }

        cl_ulong localSize() const {
            return getInfo<cl_ulong>(clGetDeviceInfo, id, CL_DEVICE_LOCAL_MEM_SIZE);
        }

        explicit operator bool() const {
            return id == nullptr;
        }

    };

    template<typename CL_OBJECT_TYPE, typename RELEASE_FUNC>
    class WrapperRAII {
        CL_OBJECT_TYPE cl_object = nullptr;
        RELEASE_FUNC* releaseFunc;
    public:
        template <typename CREATE_FUNC, typename... Args>
        WrapperRAII(CREATE_FUNC createFunc, RELEASE_FUNC releaseFunc, Args... args): releaseFunc(releaseFunc) {
            cl_int errcode_ret = 0;
            cl_object = createFunc(args..., &errcode_ret);
            OCL_SAFE_CALL(errcode_ret);
        }

        const CL_OBJECT_TYPE& getObject() const {
            return cl_object;
        }

        ~WrapperRAII() {
            if (cl_object)
                releaseFunc(cl_object);
        }
    };
    template<typename CREATE_FUNC, typename RELEASE_FUNC, typename... Args>
    auto makeWrapper(CREATE_FUNC createFunc, RELEASE_FUNC* releaseFunc, Args... args) -> WrapperRAII<decltype(createFunc(args..., nullptr)), RELEASE_FUNC> {
        return WrapperRAII<decltype(createFunc(args..., nullptr)), RELEASE_FUNC>(createFunc, releaseFunc, args...);
    }

    class KernelsInProgram {
        std::vector<cl_kernel> kernels;
    public:
        KernelsInProgram(cl_program program) {
            kernels = getInfoVec<cl_kernel, cl_uint>(clCreateKernelsInProgram, program);
        }
        std::vector<cl_kernel> getKernels() const {
            return kernels;
        }
        ~KernelsInProgram() {
            for(cl_kernel kernel: kernels) {
                clReleaseKernel(kernel);
            }
        }
    };
}