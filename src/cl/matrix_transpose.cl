#define TILE_SIZE 32

__kernel void matrix_transpose(__global const float* a, __global float* at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0), j = get_global_id(1);
    int local_i = get_local_id(0), local_j = get_local_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE];

    if (j < m && i < k) {
        tile[local_i][local_j] = a[j * k + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int i1 = i % TILE_SIZE, j1 = j % TILE_SIZE;
    int i0 = i - i1, j0 = j - j1;
    if (i0 + j1 < k && j0 + i1 < m) {
        at[(i0 + j1) * m + j0 + i1] = tile[local_j][local_i];
    }
}