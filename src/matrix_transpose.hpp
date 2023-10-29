#pragma once
//
// Created by Nikolai on 26.10.2023.
//
#include "cl/matrix_transpose_cl.h"

#include <libutils/misc.h>

#include <algorithm>

class Transpose {
private:
    ocl::Kernel matrix_transpose_kernel_;
    const unsigned int work_group_size = 16;

public:
    Transpose():
        matrix_transpose_kernel_(matrix_transpose_kernel, matrix_transpose_kernel_length, "matrix_transpose")
    {
        matrix_transpose_kernel_.compile();
    }

    template<typename T>
    void transpose(unsigned int K, unsigned int M, const T& as_gpu, T& bs_gpu)
    {
        unsigned int work_group_size_K = std::min(K, work_group_size);
        unsigned int work_group_size_M = std::min(M, work_group_size);
        matrix_transpose_kernel_.exec(gpu::WorkSize(work_group_size_K, work_group_size_M, K, M),
                                     as_gpu, bs_gpu, M, K);
    }
};
