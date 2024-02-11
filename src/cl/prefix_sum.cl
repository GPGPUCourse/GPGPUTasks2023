#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void reduce(__global const unsigned int *bs,
                     __global unsigned int *as,
                     const unsigned int n) {
    int idx = get_global_id(0);
    if (idx < n)
        as[idx] = bs[idx * 2] + bs[idx * 2 + 1];
}

__kernel void prefix_sum(__global const unsigned int *as,
                         __global unsigned int *res,
                         unsigned int stride, 
                         const unsigned int n) {
    int idx = get_global_id(0);
    
    if (idx < n && ++idx & stride)
        res[idx - 1] += as[idx / stride - 1];
}