__kernel void radix(__global unsigned int *as) {
    // TODO
}


#define MASKED_NUM(num, shift, k) ((num) >> (shift)) & ((1 << (k)) - 1)

#ifndef WG_SIZE
    #error
#endif

#ifndef BITS_COUNT
    #error
#endif

__kernel void calc_counters(__global const uint *as, uint shift, __global uint *counters) {
    uint m = MASKED_NUM(as[get_global_id(0)], shift, BITS_COUNT);
    uint wg = get_group_id(0);
    uint lid = get_local_id(0);

    __local uint local_counters[1 << BITS_COUNT];

    if(lid < 1 << BITS_COUNT)
        local_counters[lid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_add(&local_counters[m], 1);
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if(lid < 1 << BITS_COUNT)
        counters[wg * (1 << BITS_COUNT) + lid] = local_counters[lid];
}

__kernel void radix_sort(__global const uint *as, uint shift, __global const uint *counters,
                         __global const uint *sums_t, __global uint *res) {
    uint lid = get_local_id(0);
    uint num = as[get_global_id(0)];
    uint m = MASKED_NUM(num, shift,BITS_COUNT);
    uint wg = get_group_id(0);
    uint wg_count = get_num_groups(0);
    uint m_width = 1 << BITS_COUNT;
    uint prev_out = m == 0 && wg == 0 ? 0 : sums_t[m * wg_count + wg - 1];

    __local uint local_counters[1 << BITS_COUNT];

    if(lid < 1 << BITS_COUNT)
        local_counters[lid] = counters[wg * (1 << BITS_COUNT) + lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if(lid == 0) {
        for(int i = 1; i < 1 << BITS_COUNT; i++)
            local_counters[i] += local_counters[i-1];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    uint prev_in = m == 0 ? 0 : local_counters[m-1];
    uint res_idx = prev_out + lid - prev_in;
    res[res_idx] = num;
}
