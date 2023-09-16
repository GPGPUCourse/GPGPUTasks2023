//
// Created by Mi on 16.09.2023.
//

#ifndef CL_WRAPPERS_BUFFER_H
#define CL_WRAPPERS_BUFFER_H

#include "common.h"

namespace ClWrappers {

    class Buffer {
    private:
        cl_mem Memory_ = nullptr;

    public:
        Buffer(cl_context context,
               cl_mem_flags flags,
               size_t size,
               void *host_ptr);

        const cl_mem &GetMemory() const;

        operator cl_mem() const;

        ~Buffer();
    };
}

#endif //CL_WRAPPERS_BUFFER_H
