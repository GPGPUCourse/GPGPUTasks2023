//
// Created by razerford on 12.09.23.
//

#ifndef APLUSB_HELPERS_H
#define APLUSB_HELPERS_H

#include <CL/cl.h>

#include "error_handler.h"

namespace helpers {
    cl_device_id selectDevice() {
        cl_uint numberPlatforms = 0;
        eh::OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &numberPlatforms));

        std::vector<cl_platform_id> platforms(numberPlatforms);
        eh::OCL_SAFE_CALL(clGetPlatformIDs(numberPlatforms, platforms.data(), nullptr));

        cl_device_id selectedDevice = nullptr;

        for (cl_uint i = 0; i < numberPlatforms; i++) {
            cl_platform_id platformId = platforms[i];

            cl_uint numberGPU = 0;

            eh::OCL_SAFE_CALL_IGNORE(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, 0, nullptr, &numberGPU),
                                     CL_DEVICE_NOT_FOUND);

            if (numberGPU != 0) {
                std::vector<cl_device_id> devices(numberGPU);

                eh::OCL_SAFE_CALL(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, numberGPU, devices.data(), nullptr));

                selectedDevice = devices[0];
                break;
            }
        }
        if (selectedDevice == nullptr) {
            cl_platform_id platformId = platforms[0];
            cl_uint numberDevices = 0;

            eh::OCL_SAFE_CALL(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL, 0, nullptr, &numberDevices));

            std::vector<cl_device_id> devices(numberDevices);

            eh::OCL_SAFE_CALL(clGetDeviceIDs(platformId, CL_DEVICE_TYPE_GPU, numberDevices, devices.data(), nullptr));
            selectedDevice = devices[0];
        }
        return selectedDevice;
    }

    const char *getType(const cl_device_type &deviceType) {
        switch (deviceType) {
            case CL_DEVICE_TYPE_CPU:
                return "CPU";
            case CL_DEVICE_TYPE_GPU:
                return "GPU";
            default:
                return "Other";
        }
    }

    template<typename R>
    R getInfo(const cl_device_id &device, const cl_device_info &info) {
        R value;
        eh::OCL_SAFE_CALL(clGetDeviceInfo(device, info, sizeof(R), &value, nullptr));

        return value;
    }

    template<>
    std::vector<unsigned char>
    getInfo<std::vector<unsigned char>>(const cl_device_id &device, const cl_device_info &info) {
        size_t size = 0;
        eh::OCL_SAFE_CALL(clGetDeviceInfo(device, info, 0, nullptr, &size));

        std::vector<unsigned char> deviceInfo(size, 0);
        eh::OCL_SAFE_CALL(clGetDeviceInfo(device, info, size, deviceInfo.data(), nullptr));

        return deviceInfo;
    }

    void prettyPrintSelectedDevice(cl_device_id device) {
        std::cout << "*** Selected device ***" << std::endl;
        std::cout << "Name: " << getInfo<std::vector<unsigned char>>(device, CL_DEVICE_NAME).data() << std::endl;
        std::cout << "Type: " << getType(getInfo<cl_device_type>(device, CL_DEVICE_TYPE)) << std::endl;
        std::cout << "Version: " << getInfo<std::vector<unsigned char>>(device, CL_DEVICE_VERSION).data() << std::endl;
    }
}

#endif //APLUSB_HELPERS_H
