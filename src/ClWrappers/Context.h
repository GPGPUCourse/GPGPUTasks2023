//
// Created by Mi on 16.09.2023.
//

#ifndef CL_WRAPPERS_CONTEXT_H
#define CL_WRAPPERS_CONTEXT_H

#include "common.h"

namespace ClWrappers {

    class Context {
        cl_context Context_ = nullptr;

    public:
        explicit Context(cl_context &&context);

        Context(const cl_context_properties *properties,
                cl_uint num_devices,
                const cl_device_id *devices,
                void (CL_CALLBACK *pfn_notify)(
                        const char *errinfo,
                        const void *private_info, size_t cb,
                        void *user_data
                ),
                void *user_data);

        Context(const Context &context) = delete;

        Context &operator=(const Context &context) = delete;

        const cl_context &GetContext() const;

        operator cl_context() const;

        ~Context();
    };
}

#endif //CL_WRAPPERS_CONTEXT_H
