#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 7

#define WORKGROUP_SIZE 128

__kernel void bitonic(__global float *as, const unsigned int size, const unsigned int block_size) {
    // TODO
    unsigned int global_id = get_global_id(0);
    if (global_id * 2 > size) {
        return;
    }
    unsigned int block_id = 2 * global_id / block_size;
    bool block_type = block_id % 2 == 0;

    unsigned int local_block_size = block_size;

    while (local_block_size >= 2) {
        if (block_type) {
            if (as[block_id * local_block_size] > as[block_id * local_block_size + local_block_size / 2]){
                float tmp = as[block_id * local_block_size];
                as[block_id * local_block_size] = as[block_id * local_block_size + local_block_size / 2];
                as[block_id * local_block_size + local_block_size / 2] = tmp;
            }
        } else {
            if (as[block_id * local_block_size] < as[block_id * local_block_size + local_block_size / 2]){
                float tmp = as[block_id * local_block_size];
                as[block_id * local_block_size] = as[block_id * local_block_size + local_block_size / 2];
                as[block_id * local_block_size + local_block_size / 2] = tmp;
            }
        }
        local_block_size /= 2;
        block_id = 2 * global_id / local_block_size;
        barrier(CLK_GLOBAL_MEM_FENCE);
    }
}
