#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define TILE_SIZE 16
__kernel void matrix_transpose(__global const float *a, __global float *at, unsigned int m, unsigned int k)
{
    __local float tile[TILE_SIZE][TILE_SIZE];

    int i = get_global_id(0);
    int j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j][(local_i + local_j) % TILE_SIZE] = a[j * k + i]; // To solve bank conflict

    barrier(CLK_LOCAL_MEM_FENCE);

    at[i * m + j] = tile[local_j][(local_i + local_j) % TILE_SIZE];
}
