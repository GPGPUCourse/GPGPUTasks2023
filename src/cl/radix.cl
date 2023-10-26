__kernel void radix(__global unsigned int *as) {
    // TODO
}


#define MASKED_NUM(num, shift, k) ((num) >> (shift)) & ((1 << (k)) - 1)

#ifndef WG_SIZE
    #error
#endif
__kernel void calc_counters(__global const uint *as, uint shift, uint k, __global uint *counters) {
    uint m = MASKED_NUM(as[get_global_id(0)], shift, k);
    uint wg = get_group_id(0);
    uint lid = get_local_id(0);

    //assert WG_SIZE == 2^k
    __local uint local_counters[WG_SIZE];
    local_counters[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_add(&local_counters[m], 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    counters[wg * WG_SIZE + lid] = local_counters[lid];
}

__kernel void radix_sort(__global const uint *as, uint shift, uint k, __global const uint *sums,
                         __global const uint *sums_t, __global uint *res) {
    uint i = get_local_id(0);
    uint num = as[get_global_id(0)];
    uint m = MASKED_NUM(num, shift, k);
    uint wg = get_group_id(0);
    uint wg_count = get_num_groups(0);
    uint m_width = 1 << k;
    uint prev_out = m == 0 && wg == 0 ? 0 : sums_t[m * wg_count + wg - 1];
    uint prev_in = m == 0 ? 0 : ((sums[wg * m_width + (m - 1)]) - (wg == 0 ? 0 : sums[wg * m_width - 1]));
    uint res_idx = prev_out + i - prev_in;
    res[res_idx] = num;
}
