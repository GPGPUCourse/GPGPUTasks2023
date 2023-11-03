#define WORKGROUP_SIZE 64
#define BIT_STEP 4

__kernel void radix_count(__global unsigned int *as, 
                    __global unsigned int *cnt,
                    unsigned int b) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_local_id(0);
    unsigned int g = get_group_id(0);

    __local unsigned int cnt_buf[1 << BIT_STEP];

    if (j < (1 << BIT_STEP))
        cnt_buf[j] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_add(cnt_buf + ((as[i] >> b) & ((1 << BIT_STEP) - 1)), 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (j < (1 << BIT_STEP))
        cnt[(g << BIT_STEP) + j] = cnt_buf[j];
}

#define TILE_SIZE 16

__kernel void matrix_transpose(__global float* a, __global float* at, unsigned int m, unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE];
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    tile[local_j][local_i] = a[j * m + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    at[((i - local_i) + local_j) * k + (j - local_j) + local_i] = tile[local_i][local_j];
}

__kernel void prefix_sum_up(__global unsigned int *as, unsigned int n, unsigned int d) {
    unsigned int gid = get_global_id(0);
    unsigned int k = (gid << (d + 1));
    if (k + (1 << (d + 1)) - 1 >= n) return;
    as[k + (1 << (d + 1)) - 1] += as[k + (1 << d) - 1];
}

__kernel void prefix_sum_down(__global unsigned int *as, unsigned int n, unsigned int d) {
    unsigned int gid = get_global_id(0);
    unsigned int k = (gid << (d + 1));
    if (k + (1 << (d + 1)) - 1 >= n) return;
    unsigned int tmp = as[k + (1 << d) - 1];
    as[k + (1 << d) - 1] = as[k + (1 << (d + 1)) - 1];
    as[k + (1 << (d + 1)) - 1] += tmp;
}

__kernel void set_0_as_zero(__global unsigned int *as, unsigned int n) {
    as[n - 1] = 0;
}

__kernel void radix_sort(__global unsigned int *as, 
                         __global unsigned int *bs,
                         __global unsigned int *cnt,
                         unsigned int b,
                         unsigned int gc) {
    unsigned int i = get_global_id(0);
    unsigned int j = get_local_id(0);
    unsigned int g = get_group_id(0);
    
    unsigned int bits = ((as[i] >> b) & ((1 << BIT_STEP) - 1));

    unsigned int offset = cnt[gc * bits + g];
    for (int k = i - j; k < i; k++) {
        unsigned int a_bits = ((as[k] >> b) & ((1 << BIT_STEP) - 1));
        offset += (a_bits == bits);
    }

    bs[offset] = as[i];
}
