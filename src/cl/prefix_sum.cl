#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

__kernel void prefix_sum_scan(__global unsigned int *as, int n, int w) {
    int w_half = w >> 1;
    int gid = get_global_id(0);
    int lid = gid % w_half;
    int index = gid / w_half;

    int to = w_half + index * w;
    int from = to - 1;
    to += lid;
    if (to < n) {
        as[to] += as[from];
    }
}
