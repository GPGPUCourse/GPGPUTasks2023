//
// Created by Mi on 16.09.2023.
//

#ifndef CL_WRAPPERS_COMMANDQUEUE_H
#define CL_WRAPPERS_COMMANDQUEUE_H

#include "common.h"

namespace ClWrappers {

    class CommandQueue {
    private:
        cl_command_queue CommandQueue_ = nullptr;

    public:
        CommandQueue(cl_context context,
                     cl_device_id device,
                     cl_command_queue_properties properties = 0);

        CommandQueue(const CommandQueue &commandQueue) = delete;

        CommandQueue &operator=(const CommandQueue &commandQueue) = delete;

        const cl_command_queue &GetCommandQueue() const;

        operator cl_command_queue() const;

        ~CommandQueue();
    };
}

#endif //CL_WRAPPERS_COMMANDQUEUE_H
