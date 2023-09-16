//
// Created by nik on 14.09.23.
//

#ifndef GPGPUTASKS2023_DEVICE_H
#define GPGPUTASKS2023_DEVICE_H

#include "common.h"
#include <ostream>

namespace ClWrappers {

    class Device {
    private:
        cl_device_id DeviceId_ = 0;

        template<class R>
        R GetInfo(cl_device_info info) const;

    public:
        Device() = default;

        Device(cl_device_id deviceId);

        std::string GetName() const;

        const cl_device_id &GetDeviceId() const;

        operator cl_device_id() const;

        cl_device_type GetDeviceType() const;

        cl_ulong GetGlobalMemorySize() const;

        cl_ulong GetGlobalMemoryCacheSize() const;

        cl_uint GetGlobalMemoryCacheLineSize() const;

        cl_ulong GetMaxMemoryAllocSize() const;

        uint64_t GetMaxClockFrequency() const;

        cl_uint GetMaxComputeUnits() const;

        std::vector<size_t> GetWorkItemSizes() const;


        std::string GetDeviceInfo() const;
    };
}

#endif //GPGPUTASKS2023_DEVICE_H
