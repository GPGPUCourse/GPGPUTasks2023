#define TILE_SIZE 16

__kernel void matrix_transpose(__global const unsigned int *a,
                               __global unsigned int *at,
                               unsigned int n,
                               unsigned int m)
{
    __local unsigned int tile[TILE_SIZE][TILE_SIZE];

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

#define BIT_IN_BLOCK   4
#define COUNTER_Y (1 << BIT_IN_BLOCK)

int get_val(unsigned int val, unsigned int block_number) {
    return (val >> (block_number * BIT_IN_BLOCK)) & (COUNTER_Y - 1);
}

__kernel void radix_count(__global unsigned int *as,
                          __global unsigned int *counter,
                          unsigned int block_number) {
    __local unsigned int counter_row[COUNTER_Y];

    int id = get_global_id(0);
    int y = get_local_id(0);
    int x = get_group_id(0);

    if (y < COUNTER_Y) {
        counter_row[y] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_inc(&counter_row[get_val(as[id], block_number)]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (y < COUNTER_Y) {
        counter[x * COUNTER_Y + y] = counter_row[y];
    }
}

__kernel void radix_sort(__global const unsigned int * as,
                         __global const unsigned int * counter_prefix,
                         __global unsigned int * res,
                         unsigned int counter_x,
                         unsigned int block_number) {
    int id = get_global_id(0);
    int y = get_local_id(0);
    int x = get_group_id(0);
    int wgs = get_local_size(0);

    unsigned int val = get_val(as[id], block_number);
    unsigned int offset = counter_prefix[val * counter_x + x];

    for (int i = 0; i < y; ++i) {
        unsigned int d = get_val(as[x * wgs + i], block_number);
        if (d == val) {
            ++offset;
        }
    }

    res[offset] = as[id];
}
