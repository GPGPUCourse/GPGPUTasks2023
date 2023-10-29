#pragma once
//
// Created by Nikolai on 26.10.2023.
//
#include "cl/merge_sort_cl.h"

#include <libutils/misc.h>

#include <algorithm>

class MergeSort {
private:
    ocl::Kernel merge_sort_kernel_;

public:
    MergeSort() :
        merge_sort_kernel_(merge_sort_kernel, merge_sort_kernel_length, "merge_sort")
    {
        merge_sort_kernel_.compile();
    }

    template<typename T>
    void merge_sort(unsigned int n, T& as_gpu, T& bs_gpu, unsigned int shift, unsigned int mask, size_t border = 0)
    {
        if (border == 0) {
            border = n;
        }

        unsigned int workGroupSize = 128;
        workGroupSize = std::min(n, workGroupSize);
        const unsigned int global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        for (unsigned int w = 1; w < border; w <<= 1) {
            merge_sort_kernel_.exec(gpu::WorkSize(workGroupSize, global_work_size),
                       as_gpu, bs_gpu, n, w, shift, mask);
            as_gpu.swap(bs_gpu);
        }
    }
};
