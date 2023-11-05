#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j][local_i] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    int group_i = get_group_id(0);
    int group_j = get_group_id(1);

    at[group_i * m * TILE_SIZE + group_j * TILE_SIZE + local_j * m + local_i] = tile[local_i][local_j];
}
