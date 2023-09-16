//
// Created by Mi on 16.09.2023.
//

#ifndef CL_WRAPPERS_PROGRAM_H
#define CL_WRAPPERS_PROGRAM_H

#include "common.h"

namespace ClWrappers {

    class Program {
    private:
        cl_program Program_ = nullptr;

    public:
        Program(cl_context context,
                cl_uint count,
                const char **strings,
                const size_t *lengths);

        const cl_program &GetProgram() const;

        operator cl_program() const;

        void Build(cl_uint num_devices,
                   const cl_device_id *device_list,
                   const char *options,
                   void (CL_CALLBACK *pfn_notify)(cl_program program, void *user_data) = nullptr,
                   void *user_data = nullptr) const;

        std::string GetBuildLog(cl_device_id device) const;

        ~Program();
    };
}

#endif //CL_WRAPPERS_PROGRAM_H
