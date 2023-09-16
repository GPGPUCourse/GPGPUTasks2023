//
// Created by Mi on 16.09.2023.
//

#include "Program.h"

using namespace ClWrappers;

Program::Program(cl_context context, cl_uint count, const char **strings, const size_t *lengths) {
    auto errorCode = CL_SUCCESS;
    Program_ = clCreateProgramWithSource(context, count, strings, lengths, &errorCode);
    OCL_SAFE_CALL(errorCode);
}

const cl_program &Program::GetProgram() const {
    return Program_;
}

Program::operator cl_program() const {
    return GetProgram();
}

Program::~Program() {
    if (Program_) {
        OCL_SAFE_CALL(clReleaseProgram(Program_));
    }
}

void Program::Build(cl_uint num_devices, const cl_device_id *device_list, const char *options,
                    void (CL_CALLBACK *pfn_notify)(cl_program, void *), void *user_data) const {
    OCL_SAFE_CALL(clBuildProgram(Program_, num_devices, device_list, options, pfn_notify, user_data));
}

std::string Program::GetBuildLog(cl_device_id device) const {
    std::vector<char> log = GetVector<char>(clGetProgramBuildInfo, Program_, device, CL_PROGRAM_BUILD_LOG);
    return {log.begin(), log.end()};
}
