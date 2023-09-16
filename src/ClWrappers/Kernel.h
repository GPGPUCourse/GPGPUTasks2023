//
// Created by Mi on 16.09.2023.
//

#ifndef CL_WRAPPERS_KERNEL_H
#define CL_WRAPPERS_KERNEL_H

#include "common.h"

namespace ClWrappers {

    class Kernel {
    private:
        cl_kernel Kernel_ = nullptr;

    public:
        Kernel(cl_program program, const char *kernel_name);

        const cl_kernel &GetKernel() const;

        operator cl_kernel() const;

        template<class Arg>
        void SetArg(cl_uint arg_index, Arg arg) const {
            OCL_SAFE_CALL(clSetKernelArg(Kernel_, arg_index, sizeof(Arg), &arg));
        }

        ~Kernel();
    };
}

#endif //CL_WRAPPERS_KERNEL_H
