#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#include <libgpu/opencl/cl/common.cl>
#endif

#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float* src, __global float* dst, const unsigned int M, const unsigned int K)
{
    __local float lds_tile[TILE_SIZE][TILE_SIZE + 1];

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);
    const unsigned int group_i = get_group_id(0);
    const unsigned int group_j = get_group_id(1);
    if (i < K && j < M)
        lds_tile[local_j][local_i] = src[j * K + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (group_i * TILE_SIZE + local_j < K && group_j * TILE_SIZE + local_i < M)
        dst[(group_i * TILE_SIZE + local_j) * M + group_j * TILE_SIZE + local_i] = lds_tile[local_i][local_j];
}
