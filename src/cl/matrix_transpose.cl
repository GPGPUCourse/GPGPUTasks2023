#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 6

#define WORKGROUP_SIZE 16

__kernel void matrix_transpose(__global const float *input,
                               __global float *output,
                               const int M,
                               const int K)
{
    const unsigned int lid_x = get_local_id(0);
    const unsigned int lid_y = get_local_id(1);
    const unsigned int gid_x = get_global_id(0);
    const unsigned int gid_y = get_global_id(1);
    const unsigned int wid_x = get_group_id(0);
    const unsigned int wid_y = get_group_id(1);

    __local float buffer[WORKGROUP_SIZE * WORKGROUP_SIZE];
    buffer[lid_y * WORKGROUP_SIZE + (lid_x + lid_y) % WORKGROUP_SIZE] = input[gid_y * M + gid_x];

    barrier(CLK_LOCAL_MEM_FENCE);
    const unsigned int index_x = wid_y * WORKGROUP_SIZE + lid_x;
    const unsigned int index_y = wid_x * WORKGROUP_SIZE + lid_y;

    output[index_y * K + index_x] = buffer[lid_x * WORKGROUP_SIZE + (lid_y + lid_x) % WORKGROUP_SIZE];
}