//
// Created by Mi on 16.09.2023.
//

#include "Buffer.h"

using namespace ClWrappers;

Buffer::Buffer(cl_context context, cl_mem_flags flags, size_t size, void *host_ptr) {
    auto errorCode = CL_SUCCESS;
    Memory_ = clCreateBuffer(context, flags, size, host_ptr, &errorCode);
    OCL_SAFE_CALL(errorCode);
}

const cl_mem& Buffer::GetMemory() const {
    return Memory_;
}

Buffer::operator cl_mem() const {
    return Memory_;
}

Buffer::~Buffer() {
    if (Memory_) {
        OCL_SAFE_CALL(clReleaseMemObject(Memory_));
    }
}
