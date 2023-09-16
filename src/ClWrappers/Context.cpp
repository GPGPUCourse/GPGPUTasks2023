//
// Created by Mi on 16.09.2023.
//

#include "Context.h"

using namespace ClWrappers;

Context::Context(cl_context&& context) : Context_(context)
{ }

Context::Context(const cl_context_properties *properties,
                 cl_uint num_devices,
                 const cl_device_id *devices,
                 void (CL_CALLBACK *pfn_notify)(const char *, const void *, size_t, void *),
                 void *user_data) {
    cl_int errorCode = CL_SUCCESS;
    Context_ = clCreateContext(properties, num_devices, devices, pfn_notify, user_data, &errorCode);
    OCL_SAFE_CALL(errorCode);
}

const cl_context& Context::GetContext() const {
    return Context_;
}

Context::operator cl_context() const {
    return Context_;
}

Context::~Context() {
    if (Context_) {
        OCL_SAFE_CALL(clReleaseContext(Context_));
    }
}
