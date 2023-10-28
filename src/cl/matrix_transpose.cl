#define TILE_SIZE 16

__kernel void matrix_transpose(
    __global const float *matrix,
    __global float *matrixT,
    unsigned M,
    unsigned K) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    int local_col = get_local_id(0);
    int local_row = get_local_id(1);
    __local float tile[TILE_SIZE + 1][TILE_SIZE];
    if (row < M && col < K)
        tile[local_row][local_col] = matrix[row * K + col];
    barrier(CLK_LOCAL_MEM_FENCE);
#define new_local_row local_col
#define new_local_col local_row
    int new_col = (get_group_id(0) * TILE_SIZE) + new_local_col;
    int new_row = (get_group_id(1) * TILE_SIZE) + new_local_row;
    if (new_row < M && new_col < K)
        matrixT[new_col * M + new_row] = tile[new_local_row][new_local_col];
}

__kernel void matrix_transpose_naive(
    __global const float *matrix,
    __global float *matrixT,
    unsigned M,
    unsigned K) {
    int col = get_global_id(0);
    int row = get_global_id(1);
    if (col < K && row < M)
        matrixT[col * M + row] = matrix[row * K + col];
}
