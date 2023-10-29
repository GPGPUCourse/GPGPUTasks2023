#pragma once
//
// Created by Nikolai on 26.10.2023.
//
#include "cl/prefix_sum_cl.h"

#include <libutils/misc.h>

class PrefixSum {
private:
    ocl::Kernel prefix_sum_scan_;
    ocl::Kernel prefix_sum_map_;
    ocl::Kernel prefix_sum_reduce_;

public:
    PrefixSum() :
        prefix_sum_scan_(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum_scan"),
        prefix_sum_map_(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum_map"),
        prefix_sum_reduce_(prefix_sum_kernel, prefix_sum_kernel_length, "prefix_sum_reduce")
    {
        prefix_sum_scan_.compile();
        prefix_sum_map_.compile();
        prefix_sum_reduce_.compile();
    }

    template<typename T>
    void prefix_sum_scan(size_t n, T& as_gpu)
    // as_gpu.resizeN(n);
    // as_gpu.writeN(as.data(), n);
    // as_gpu.readN(res.data(), n);
    {
        unsigned int workGroupSize = 128;
        unsigned int global_work_size = (n / 2 + workGroupSize - 1) / workGroupSize * workGroupSize;

        for (int width = 2; width < 2 * n; width <<= 1) {
            prefix_sum_scan_.exec(gpu::WorkSize(workGroupSize, global_work_size),
                                 as_gpu, n, width);
        }
    }

    template<typename T>
    void prefix_sum(size_t n, T& as_gpu)
    // as_gpu.resizeN(n + 1);
    // as_gpu.writeN(as.data(), n + 1); // нужно занулить последний элемент для многократного исполнения ядра
    // as_gpu.readN(res.data(), n, 1);
    {
        unsigned int workGroupSize = 128;
        unsigned int global_work_size = 0;
        int width;
        for (width = 2; width < 2 * n; width <<= 1) {
            global_work_size = (n / width + workGroupSize - 1) / workGroupSize * workGroupSize;
            global_work_size = std::max(global_work_size, 1U);
            prefix_sum_map_.exec(gpu::WorkSize(std::min(workGroupSize, global_work_size), global_work_size),
                                as_gpu, n, width);
        }

        for (; width > 1; width >>= 1) {
            global_work_size = (n / width + workGroupSize - 1) / workGroupSize * workGroupSize;
            global_work_size = std::max(global_work_size, 1U);
            prefix_sum_reduce_.exec(gpu::WorkSize(std::min(workGroupSize, global_work_size), global_work_size),
                                   as_gpu, n, width);
        }
    }
};
