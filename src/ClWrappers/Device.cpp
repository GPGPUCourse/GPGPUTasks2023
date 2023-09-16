//
// Created by nik on 14.09.23.
//

#include "Device.h"

using namespace ClWrappers;

template<class R>
R Device::GetInfo(cl_device_info info) const {
    return GetValue<R>(clGetDeviceInfo, DeviceId_, info);
}

Device::Device(cl_device_id deviceId) : DeviceId_(deviceId)
{ }

const cl_device_id& Device::GetDeviceId() const {
    return DeviceId_;
}

Device::operator cl_device_id() const {
    return GetDeviceId();
}

std::string Device::GetName() const {
    auto deviceName = GetVector<char>(clGetDeviceInfo, DeviceId_, CL_DEVICE_NAME);
    return {deviceName.begin(), deviceName.end()};
}

cl_device_type Device::GetDeviceType() const {
    return GetInfo<cl_device_type>(CL_DEVICE_TYPE);
}

cl_ulong Device::GetGlobalMemorySize() const {
    return GetInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_SIZE);
}

cl_ulong Device::GetGlobalMemoryCacheSize() const {
    return GetInfo<cl_ulong>(CL_DEVICE_GLOBAL_MEM_CACHE_SIZE);
}

cl_uint Device::GetGlobalMemoryCacheLineSize() const {
    return GetInfo<cl_uint>(CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE);
}

cl_ulong Device::GetMaxMemoryAllocSize() const {
    return GetInfo<cl_ulong>(CL_DEVICE_MAX_MEM_ALLOC_SIZE);
}

uint64_t Device::GetMaxClockFrequency() const {
    return GetInfo<uint64_t>(CL_DEVICE_MAX_CLOCK_FREQUENCY);
}

cl_uint Device::GetMaxComputeUnits() const {
    return GetInfo<cl_uint>(CL_DEVICE_MAX_COMPUTE_UNITS);
}

std::vector<size_t> Device::GetWorkItemSizes() const {
    return GetVector<size_t>(clGetDeviceInfo, DeviceId_, CL_DEVICE_MAX_WORK_ITEM_SIZES);
}

std::string Device::GetDeviceInfo() const {
    std::stringstream os;
    os << "        Device(" << DeviceId_ << ")" << std::endl;
    os << "            NAME: " << GetName() << std::endl;

    os << "            TYPE: ";
    auto deviceType = GetDeviceType();
    if (deviceType & CL_DEVICE_TYPE_CPU)
        os << "CL_DEVICE_TYPE_CPU ";
    if (deviceType & CL_DEVICE_TYPE_GPU)
        os << "CL_DEVICE_TYPE_GPU ";
    if (deviceType & CL_DEVICE_TYPE_ACCELERATOR)
        os << "CL_DEVICE_TYPE_ACCELERATOR ";
    if (deviceType & CL_DEVICE_TYPE_DEFAULT)
        os << "CL_DEVICE_TYPE_DEFAULT ";
    os << std::endl;

    os << "            GLOBAL_MEM_SIZE: " << (GetGlobalMemorySize() >> 20) << " MB" << std::endl;
    os << "            GLOBAL_MEM_CACHE_SIZE: " << GetGlobalMemoryCacheSize() << " B" << std::endl;
    os << "            GLOBAL_MEM_CACHELINE_SIZE: " << GetGlobalMemoryCacheLineSize() << " B" << std::endl;
    os << "            MAX_MEM_ALLOC_SIZE: " << (GetMaxMemoryAllocSize() >> 20) << " MB" << std::endl;
    os << "            MAX_CLOCK_FREQUENCY: " << GetMaxClockFrequency() << " MHz" << std::endl;
    os << "            MAX_COMPUTE_UNITS: " << GetMaxComputeUnits() << std::endl;

    os << "            DEVICE_MAX_WORK_ITEM_SIZES: ";
    for(auto size : GetWorkItemSizes()) {
        os << size << "; ";
    }

    return os.str();
}
