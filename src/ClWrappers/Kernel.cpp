//
// Created by Mi on 16.09.2023.
//

#include "Kernel.h"

using namespace ClWrappers;

Kernel::Kernel(cl_program program, const char *kernel_name) {
    auto errorCode = CL_SUCCESS;
    Kernel_ = clCreateKernel(program, kernel_name, &errorCode);
    OCL_SAFE_CALL(errorCode);
}

const cl_kernel& Kernel::GetKernel() const {
    return Kernel_;
}

Kernel::operator cl_kernel() const {
    return GetKernel();
}

Kernel::~Kernel() {
    if (Kernel_) {
        OCL_SAFE_CALL(clReleaseKernel(Kernel_));
    }
}

