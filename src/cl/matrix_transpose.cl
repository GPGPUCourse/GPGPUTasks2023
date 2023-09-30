__kernel void matrix_transpose(__global const float *a,
                               __global float *at,
                               unsigned int n,
                               unsigned int m)
{
    __local float tile[TILE_SIZE][TILE_SIZE];

    int y = get_global_id(0);
    int x = get_global_id(1);

    int local_y = get_local_id(0);
    int local_x = get_local_id(1);

    int group_y = get_group_id(0);
    int group_x = get_group_id(1);

    tile[local_x][local_y] = a[x * m + y];
    barrier(CLK_LOCAL_MEM_FENCE);

    at[(group_y * TILE_SIZE + local_x) * n + group_x * TILE_SIZE + local_y] = tile[local_y][local_x];
}
