#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif
#line 5

__kernel void reduce(
            __global unsigned int *as,
            unsigned int block_size,
            unsigned int n
        ) {
    unsigned int gid = get_global_id(0);
    unsigned int ind = (gid + 1) * 2 * block_size - 1;
    if (ind < n) {
        as[ind] += as[ind - block_size];
    }
}

__kernel void add_to_result(
            __global unsigned int* as,
            __global unsigned int* res,
            unsigned int block_size
        ) {
    unsigned int gid = get_global_id(0);
    unsigned int tail = gid % block_size;
    unsigned int ind = (gid / block_size) * 2 * block_size + block_size + tail;
    res[ind - 1] += as[ind - tail - 1];
}