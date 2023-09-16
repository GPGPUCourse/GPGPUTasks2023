//
// Created by Mi on 16.09.2023.
//

#include "CommandQueue.h"

using namespace ClWrappers;

CommandQueue::CommandQueue(cl_context context,
                           cl_device_id device,
                           cl_command_queue_properties properties) {
    cl_int errorCode = CL_SUCCESS;
    CommandQueue_ = clCreateCommandQueue(context, device, properties, &errorCode);
    OCL_SAFE_CALL(errorCode);
}

const cl_command_queue& CommandQueue::GetCommandQueue() const {
    return CommandQueue_;
}

CommandQueue::operator cl_command_queue() const {
    return GetCommandQueue();
}

CommandQueue::~CommandQueue() {
    if (CommandQueue_) {
        OCL_SAFE_CALL(clReleaseCommandQueue(CommandQueue_));
    }
}