#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif
#line 5
// Для быстродействия будем считать, что n и work_group_size - это степени 2

#define WORK_GROUP_SIZE 256
__kernel void merge_small(__global const float* in,
                          __global float* out,
                          unsigned int block_size)
{
    unsigned int lid = get_local_id(0);
    unsigned int gid = get_global_id(0);
    __local float buf[WORK_GROUP_SIZE];
    buf[lid] = in[gid];
    barrier(CLK_LOCAL_MEM_FENCE);


    float val = buf[lid];
    unsigned int block_pair_index = lid / (2 * block_size);
    unsigned int block_pair_start = block_pair_index * 2 * block_size;
    unsigned int global_block_pair_start = gid / WORK_GROUP_SIZE * WORK_GROUP_SIZE;
    unsigned int llid = lid - block_pair_start;
    unsigned int offset = llid < block_size ? 1 : 0; // важно: внутри варпа offset не меняется
    int l = block_pair_start + offset - 1;
    unsigned int r = block_size + l;
    while (r - l > 1) {
        unsigned int m = (l + r + 1 - offset) / 2;
        if (buf[offset * block_size + m] > val) {
            r = m;
        } else if (buf[offset * block_size + m] < val) {
            l = m;
        } else {
            if (offset) {
                l = m;
            } else {
                r = m;
            }
        }
    }
    if (offset) {
        if (buf[l + block_size] > val) {
            r = l;
        }
        out[global_block_pair_start + llid + r] = val;
    } else {
        if (buf[r] < val) {
            l = r;
        }
        out[global_block_pair_start + llid - block_size + l + 1] = val;
    }
}


__kernel void calculate_inds(__global const float* in,
                             __global unsigned int* out,
                             unsigned int block_size)
{
    unsigned int gid = get_global_id(0);
    unsigned int work_groups_per_block_pair = 2 * block_size / WORK_GROUP_SIZE;
    unsigned int block_pair_index = gid / work_groups_per_block_pair;
    unsigned int local_ind_sum = WORK_GROUP_SIZE * (gid - block_pair_index * work_groups_per_block_pair + 1);
    unsigned int block_pair_start = block_pair_index * 2 * block_size;
    unsigned int l = 0;
    if (local_ind_sum > block_size) {
        l = local_ind_sum - block_size;
    }
    unsigned int r = local_ind_sum;
    if (block_size < r) {
        r = block_size;
    }
    while (r - l > 1) {
        unsigned int m = (l + r + 1) / 2;
        float a = in[block_pair_start + m - 1];
        float b = in[block_pair_start + block_size + local_ind_sum - m];
        if (a < b) {
            l = m;
        } else {
            r = m;
        }
    }
    float a = in[block_pair_start + r - 1];
    float b = in[block_pair_start + block_size + local_ind_sum - r];
    if (a < b) {
        l = r;
    }
    out[gid] = block_pair_start + l;
}

__kernel void merge_large(__global const float* in,
                          __global const unsigned int* ind,
                          __global float* out,
                          unsigned int block_size)
{
    unsigned int lid = get_local_id(0);
    unsigned int gid = get_global_id(0);
    unsigned int work_groups_per_block_pair = 2 * block_size / WORK_GROUP_SIZE;
    unsigned int block_pair_index = gid / (2 * block_size);
    unsigned int block_pair_start = block_pair_index * 2 * block_size;
    unsigned int work_group_index = get_group_id(0);
    unsigned int a_from, a_to;
    unsigned int work_group_mod = work_group_index % work_groups_per_block_pair;
    if (work_group_mod == 0) {
        a_from = block_pair_start;
    } else {
        a_from = ind[work_group_index - 1];
    }
    a_to = ind[work_group_index];
    unsigned int as = a_to - a_from;
    unsigned int bs = WORK_GROUP_SIZE - as;
    unsigned int b_from = 2 * block_pair_start + WORK_GROUP_SIZE * work_group_mod - a_from + block_size;

    __local float buf[WORK_GROUP_SIZE];
    unsigned int offset = lid < as ? 1 : 0;
    if (offset) {
        buf[lid] = in[a_from + lid];
    } else {
        buf[lid] = in[b_from + lid - as];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int l = 0;
    if (l + bs < lid) {
        l = lid - bs;
    }
    unsigned int r = lid;
    if (r > as) {
        r = as;
    }
    while (r - l > 1) {
        unsigned int m = (l + r + 1) / 2;
        if (buf[m-1] < buf[as + lid - m]) {
            l = m;
        } else {
            r = m;
        }
    }
    if (buf[r-1] < buf[as + lid - r]) {
        l = r;
    }
    unsigned int k = lid - l;
    if (l == as) {
        out[gid] = buf[as + k];
    } else if (k == bs) {
        out[gid] = buf[l];
    } else {
        float a = buf[l];
        float b = buf[as + k];
        out[gid] = a < b ? a : b;
    }
}
