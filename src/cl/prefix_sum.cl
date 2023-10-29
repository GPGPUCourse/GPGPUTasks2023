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

__kernel void prefix_sum_map(__global unsigned int *as, int n, int w) {
    int w_half = w >> 1;
    int gid = get_global_id(0);

    int from = w_half - 1 + w * gid;
    int to = from + w_half;
    if (to < n) {
        as[to] += as[from];
    }
}

__kernel void prefix_sum_reduce(__global unsigned int *as, int n, int w) {
    int w_half = w >> 1;
    int gid = get_global_id(0);

    int from = w_half - 1 + w * gid;
    int to = from + w_half;
    unsigned int to_v = 0;
    if (to < n) {
        to_v = as[to];
    } else {
        to = from + 1;
    }
    if (from < n) {
        as[to] += as[from];
        as[from] = to_v;
    }
}
