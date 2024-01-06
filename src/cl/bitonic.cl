#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 7

#define WORKGROUP_SIZE 128

__kernel void bitonic
        (
                __global float *as, const unsigned int size, const unsigned int block_size,
                const unsigned int sub_block_size
        ) {
    // TODO
    unsigned int global_id = get_global_id(0);
    if (global_id * 2 >= size) {
        return;
    }
    unsigned int block_id = 2 * global_id / block_size; // 0
    bool block_type = block_id % 2 == 0; // true

    unsigned int local_block_size = sub_block_size; // 2
    unsigned int local_block_id = 2 * global_id / sub_block_size; // 1
    unsigned int offset = global_id % (sub_block_size / 2); // 0
    if (block_type) {
        if (as[local_block_id * local_block_size + offset] > as[local_block_id * local_block_size + offset + local_block_size / 2]) {
            float tmp = as[local_block_id * local_block_size + offset];
            as[local_block_id * local_block_size + offset] = as[local_block_id * local_block_size + offset + local_block_size / 2];
            as[local_block_id * local_block_size + offset + local_block_size / 2] = tmp;
        }
    } else {
        if (as[local_block_id * local_block_size + offset] < as[local_block_id * local_block_size + offset + local_block_size / 2]) {
            float tmp = as[local_block_id * local_block_size + offset];
            as[local_block_id * local_block_size + offset] = as[local_block_id * local_block_size + offset + local_block_size / 2];
            as[local_block_id * local_block_size + offset + local_block_size / 2] = tmp;
        }
    }
}
