__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int m, unsigned int k)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE];
    int local_j = get_local_id(0);
    int local_i = get_local_id(1);
    tile[local_j][(local_i + local_j) % TILE_SIZE] = a[i * k + j]; //циклически сдвигаем каждую строку на номер строки вправо
    barrier(CLK_LOCAL_MEM_FENCE);
    at[(j - j % TILE_SIZE + local_i) * m + (i - i % TILE_SIZE + local_j)] = tile[local_i][(local_i + local_j) % TILE_SIZE];
}