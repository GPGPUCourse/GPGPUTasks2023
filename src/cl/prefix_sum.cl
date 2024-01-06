// TODO
#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 7

#define WORKGROUP_SIZE 128

__kernel void reduce(__global const unsigned int *as, __global unsigned int *bs, unsigned int size) {
    unsigned int global_id = get_global_id(0);

    unsigned int index = 2 * global_id;
    if (index < size) {
        bs[global_id] = as[index] + as[index + 1];
    }
}

__kernel void prefix_sum(__global const unsigned int *as, __global unsigned int *bs, unsigned int bs_size, unsigned int offset) {
    unsigned int global_id = get_global_id(0);
    if (global_id >= bs_size) {
        return;
    }

    unsigned int index = 1 + global_id;
    if ((index >> offset) % 2 == 1) {
        bs[global_id] += as[index / (1 << offset) - 1];
    }
}