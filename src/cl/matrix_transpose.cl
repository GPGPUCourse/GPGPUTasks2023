#line 2

#define TILE_SIZE 16
__kernel void matrix_transpose(__global const float *as, __global float *ast, unsigned int M, unsigned int K)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    __local float tile[TILE_SIZE][TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (i < M && j < K)
    {
        tile[local_i][local_j] = as[j * K + i];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    float tmp = tile[local_i][local_j];
    tile[local_i][local_j] = tile[local_j][local_i];
    tile[local_j][local_i] = tmp;
    barrier(CLK_LOCAL_MEM_FENCE);

    ast[i * M + j] = tile[local_i][local_j];
}