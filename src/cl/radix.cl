#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif
#line 5
// Для быстродействия будем считать, что n и work_group_size - это степени 2

#define WORK_GROUP_SIZE 256
#define VALUE_BYTES 2
#define KEYS 4
#define TILE_SIZE 64
unsigned int cmp_val(unsigned int val, unsigned int stage){
    return (val >> (stage * VALUE_BYTES)) & (KEYS - 1);
}

__kernel void merge_small(__global const unsigned int* in,
                          __global unsigned int* out,
                          unsigned int block_size,
                          unsigned int stage)
{
    unsigned int lid = get_local_id(0);
    unsigned int gid = get_global_id(0);
    __local unsigned int buf[WORK_GROUP_SIZE];
    buf[lid] = in[gid];
    barrier(CLK_LOCAL_MEM_FENCE);


    unsigned int val = buf[lid];
    unsigned int val_local = cmp_val(val, stage);
    unsigned int block_pair_index = lid / (2 * block_size);
    unsigned int block_pair_start = block_pair_index * 2 * block_size;
    unsigned int global_block_pair_start = gid / WORK_GROUP_SIZE * WORK_GROUP_SIZE;
    unsigned int llid = lid - block_pair_start;
    unsigned int offset = llid < block_size ? 1 : 0; // внутри варпа offset не меняется при достаточно больших блоках
    int l = block_pair_start - offset;
    unsigned int r = block_size + l;
    while (r - l > 1) {
        unsigned int m = (l + r + offset) / 2;
        unsigned int m_val = cmp_val(buf[offset * block_size + m], stage);
        if (m_val > val_local) {
            r = m;
        } else if (m_val < val_local) {
            l = m;
        } else {
            if (offset) {
                r = m;
            } else {
                l = m;
            }
        }
    }
    if (offset) {
        if (cmp_val(buf[r + block_size], stage) < val_local) {
            l = r;
        }
        out[global_block_pair_start + llid + l + 1] = val;
    } else {
        if (cmp_val(buf[l], stage) > val_local) {
            r = l;
        }
        out[global_block_pair_start + llid - block_size + r] = val;
    }
}

__kernel void get_counts(__global const unsigned int* in,
                          __global unsigned int* cs,
                          unsigned int stage)
{
    __local unsigned int counts_local[KEYS];
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int wid = get_group_id(0);
    if (lid < KEYS) {
        counts_local[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(&counts_local[cmp_val(in[gid], stage)], 1);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid < KEYS) {
        cs[wid * KEYS + lid] = counts_local[lid];
    }
}

__kernel void zero(
        __global unsigned int* in
        ) {
    in[get_global_id(0)] = 0;
}

__kernel void reduce(
        __global unsigned int *as,
        unsigned int block_size,
        unsigned int n
) {
    unsigned int gid = get_global_id(0);
    unsigned int ind = (gid + 1) * 2 * block_size - 1;
    if (ind < n) {
        as[ind] += as[ind - block_size];
    }
}

__kernel void add_to_result(
        __global unsigned int* as,
        __global unsigned int* res,
        unsigned int block_size
) {
    unsigned int gid = get_global_id(0);
    unsigned int tail = gid % block_size;
    unsigned int ind = (gid / block_size) * 2 * block_size + block_size + tail;
    res[ind - 1] += as[ind - tail - 1];
}

__kernel void matrix_transpose(__global const unsigned int* as,
                               __global unsigned int* as_t,
                               unsigned int M)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_j = get_local_id(1);
    unsigned int lid = local_j * KEYS + i;
     // int local_ij = (local_i + local_j) % KEYS;
    __local unsigned int tile[TILE_SIZE * KEYS];

    if (j < M) {
        tile[lid] = as[j * KEYS + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int dst_y = lid / TILE_SIZE;
    unsigned int dst_x = lid - dst_y * TILE_SIZE;
    if (dst_x < M) {
        as_t[dst_y * M + dst_x + j - local_j] = tile[dst_x * KEYS + dst_y];
    }
}

__kernel void local_prefix(__global const unsigned int* as,
                               __global unsigned int* ps)
{
    unsigned int gid = get_global_id(0);
    unsigned int gs = get_global_size(0);
    unsigned int sum = 0;
    for (unsigned int i=0; i<KEYS-1; i++) {
        unsigned int ind = gs * i + gid;
        sum += as[ind];
        ps[ind] = sum;
    }
}

__kernel void radix(__global const unsigned int* as,
                    __global const unsigned int* ps,
                    __global const unsigned int* psb,
                    __global unsigned int* res,
                    unsigned int stage)
{
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int wg_count = get_num_groups(0);

    unsigned int val = as[gid];
    unsigned int key = cmp_val(val, stage);
    unsigned int ind_res = lid;
    unsigned int ind_p = wg_count * key + group_id;
    if (ind_p > 0) {
        ind_res += ps[ind_p - 1];
    }
    if (key > 0) {
        ind_res -= psb[wg_count * (key - 1) + group_id];
    }
    res[ind_res] = val;
}
