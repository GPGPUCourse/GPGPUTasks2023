#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6


__kernel void prefix(__global float *as, __global float *res, const unsigned int n, const unsigned int offset) {
    int i = get_global_id(0);

    res[i] = as[i];

    int j = i - offset;

    if (j >= 0) {
        res[i] += as[i];
    }
}
