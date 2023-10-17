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

    const int local_ind_tr = (local_col * local_ncol) + local_row;
    
    const int tile_col_tr = local_ind_tr % TILE_COLS;
    const int tile_row_tr = local_ind_tr / TILE_COLS;

    const int base_col = get_group_id(1) * local_nrow;
    const int base_row = get_group_id(0) * local_ncol;

    const int ind_tr = ((base_row + local_row) * ncol) + (base_col + local_col);

    mat_tr[ind_tr] = local_mat[tile_row_tr][tile_col_tr];
}