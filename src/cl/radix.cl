#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#include <stdio.h>
#endif

#define BITS_PER_ITER 4
#define BITS_VALUE (1 << BITS_PER_ITER)

__kernel void radix_count(__global unsigned int *as, __global unsigned int* counting, unsigned int shift) {
    __local unsigned int count_local[BITS_VALUE];
    int group_id = get_group_id(0);
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);

    if (local_id < BITS_VALUE) {
        count_local[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_inc(&count_local[(as[global_id] >> shift) & (BITS_VALUE - 1)]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < BITS_VALUE) {
        counting[group_id * BITS_VALUE + local_id] = count_local[local_id];
    }
}

#define WORK_SIZE 128
__kernel void radix(__global unsigned int *as, __global unsigned int* bs, __global unsigned int* prefix_sum, __global unsigned int* cnt, unsigned int shift) {
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int group_id = get_group_id(0);
    unsigned int group_cnt = get_num_groups(0);

    __local unsigned int local_as[WORK_SIZE];

    local_as[local_id] = (as[global_id] >> shift) & (BITS_VALUE - 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int offset = 0;
    unsigned int cur_value = local_as[local_id];
    for (int i = 0; i < local_id; ++i) {
        offset += cur_value == local_as[i] ? 1 : 0;
    }
    int prev_values = prefix_sum[group_cnt * cur_value + group_id] - cnt[group_id * BITS_VALUE + cur_value];
    bs[prev_values + offset] = as[global_id];
}
