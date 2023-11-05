#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6


__kernel void get_counts(__global const unsigned int *as, __global unsigned int *res, const unsigned int offset) {
    const unsigned global_id = get_global_id(0);
    const unsigned local_id = get_local_id(0);
    const unsigned group_id = get_group_id(0);

    unsigned int cell = (as[global_id] >> offset) & 15;

    __local unsigned int local_counter[16];

    if (local_id < 16)
        local_counter[local_id] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_add(local_counter + cell, 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < 16)
        res[group_id * 16 + local_id] = local_counter[local_id];
}

__kernel void radix(__global unsigned int *as, __global unsigned int *res, __global unsigned int *pref_cnt,
                    __global unsigned int *pref_cnt_t, const unsigned int n, const unsigned int offset) {
    const unsigned global_id = get_global_id(0);
    const unsigned group_id = get_group_id(0);

    unsigned int block = (as[global_id] >> offset) & 15;
    unsigned int plus = (n / 128) * block;
    unsigned int i = plus + group_id;

    unsigned int ans_ind = global_id % 128;
    if (i > 0) {
        ans_ind += pref_cnt_t[i - 1];
    }
    if (block > 0) {
         ans_ind -= pref_cnt[group_id * 16 + block - 1];
    }

    res[ans_ind] = as[global_id];
}
