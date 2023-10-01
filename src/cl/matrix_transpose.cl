#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float* a,
                               __global float* at,
                               unsigned int m,
                               unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (j < m && i < k)
        tile[local_i][local_j] = a[j * k + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    int diff_i = i - local_i;
    int diff_j = j - local_j;

    if (diff_j + local_i < m && diff_i + local_j < k)
        at[(diff_i + local_j) * m + diff_j + local_i] = tile[local_j][local_i];
}