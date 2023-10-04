#define TILE_ROWS 16
#define TILE_COLS (TILE_ROWS + 1)
__kernel void matrix_transpose(__global const float *mat, __global float *mat_tr, int nrow, int ncol) {
    const int local_col = get_local_id(0);
    const int local_row = get_local_id(1);

    const int local_ncol = get_local_size(0);
    const int local_nrow = get_local_size(1);

    const int local_ind = (local_row * local_ncol) + local_col;

    const int col = get_global_id(0);
    const int row = get_global_id(1);

    __local float local_mat[TILE_ROWS][TILE_COLS];

    const int tile_col = local_ind % TILE_COLS;
    const int tile_row = local_ind / TILE_COLS;

    const int ind = (row * ncol) + col;

    local_mat[tile_row][tile_col] = mat[ind];

    barrier(CLK_LOCAL_MEM_FENCE);

    const int ind_tr = (col * nrow) + row;

    mat_tr[ind_tr] = local_mat[tile_row][tile_col];
}