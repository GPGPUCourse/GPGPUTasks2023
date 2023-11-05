#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6


__kernel void prefix(__global unsigned int *as, __global unsigned int *res, const unsigned int offset) {
    unsigned int i = get_global_id(0);

    res[i] = as[i];

    int j = i - offset;

    if (j >= 0) {
        res[i] += as[j];
    }
}

__kernel void prefix_row(
        __global const unsigned int *as,
        __global unsigned int *res)
{
    const int global_id = get_global_id(0);

    if (global_id % 16 == 0) {
        unsigned accum = 0;
        for (int i = global_id; i < global_id + 16; i++) {
            accum += as[i];
            res[i] = accum;
        }
    }
}
