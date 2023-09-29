#define TILE_SIZE 32
__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int m, unsigned int k) {
    // TODO
    __local float tile[TILE_SIZE][TILE_SIZE + 1];

    const uint local_i = get_local_id(0);
    const uint local_j = get_local_id(1);

    const uint group_i = get_group_id(0);
    const uint group_j = get_group_id(1);

    const uint local_size_i = get_local_size(0);
    const uint local_size_j = get_local_size(1);

    int xIndex = group_i * local_size_i + local_i;
    int yIndex = group_j * local_size_j + local_j;

    if ((xIndex < k) && (yIndex < m)) {
        int idx = yIndex * m + xIndex;
        tile[local_j][local_i] = a[idx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    xIndex = group_j * local_size_j + local_i;
    yIndex = group_i * local_size_i + local_j;

    if ((xIndex < m) && (yIndex < k)) {
        int idx = yIndex * k + xIndex;
        at[idx] = tile[local_i][local_j];
    }
}
