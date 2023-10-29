#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

__kernel void prefix_sum(__global unsigned int* as, __global unsigned int* res, unsigned int k) {
    unsigned gid = get_global_id(0);
    if (gid >= k) {
        res[gid] = as[gid] + as[gid - k];
    } else
        res[gid] = as[gid];
}