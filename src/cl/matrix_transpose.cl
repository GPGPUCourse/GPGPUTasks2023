#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float *matrix,
                               __global float *matrixTransposed,
                               const unsigned M,
                               const unsigned K) {

    unsigned col = get_global_id(0);
    unsigned row = get_global_id(1);
    
    unsigned localCol = get_local_id(0);
    unsigned localRow = get_local_id(1);
    
    __local float buff[TILE_SIZE + 1][TILE_SIZE];
    buff[localRow][localCol] = matrix[row * M + col];

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned tCol = get_group_id(0) * TILE_SIZE + localRow;
    unsigned tRow = get_group_id(1) * TILE_SIZE + localCol;
    matrixTransposed[tCol * M + tRow] = buff[localCol][localRow];
}