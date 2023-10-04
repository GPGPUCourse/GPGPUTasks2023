#define TILE_SIZE 16
__kernel void matrix_transpose(__global const float *mat, __global float *mat_tr, int nrow, int ncol) {
    const int local_col = get_local_id(0);
    const int local_row = get_local_id(1);

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    __local float local_mat[TILE_SIZE][TILE_SIZE];

    const int ind = (row * ncol) + col;

    local_mat[local_row][local_col] = mat[ind];

    barrier(CLK_LOCAL_MEM_FENCE);

    const int ind_tr = (col * nrow) + row;

    mat_tr[ind_tr] = local_mat[local_row][local_col];
}