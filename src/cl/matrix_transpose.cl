#line 2

#define TILE_SIZE 16
__kernel void matrix_transpose(__global const float *as, __global float *ast, unsigned int M, unsigned int K)
{
    int i = get_global_id(0);  // 0 to K
    int j = get_global_id(1);  // 0 to M

    __local float tile[TILE_SIZE][TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (i < K && j < M)
    {
        tile[local_j][local_i] = as[j * K + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < K && j < M)
    {
        ast[i * M + j] = tile[local_j][local_i];
    }
}

__kernel void matrix_transpose_banks(__global const float *as, __global float *ast, unsigned int M, unsigned int K)
{
    int i = get_global_id(0);  // 0 to K
    int j = get_global_id(1);  // 0 to M

    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (i < K && j < M)
    {
        tile[local_j][local_i] = as[j * K + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < K && j < M)
    {
        ast[i * M + j] = tile[local_j][local_i];
    }
}