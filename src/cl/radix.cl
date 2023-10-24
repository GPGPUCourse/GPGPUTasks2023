__kernel void reduce(__global const unsigned int *as, __global unsigned int *bs, unsigned int n, unsigned int i) {
    const unsigned int global_i = get_global_id(0);
    if (global_i < n && ((((global_i + 1) >> i) & 1))) {
        bs[global_i] += as[((global_i + 1) >> i) - 1];
    }
}

__kernel void prefix_sum(__global const unsigned int *as, __global unsigned int *res, unsigned int n) {
    const unsigned int i = get_global_id(0);
    if (i < n) {
        res[i] = as[i * 2] + as[i * 2 + 1];
    }
}

#define SEGMENT_SIZE 128
__kernel void radix_counters(__global const unsigned int *as, __global unsigned int *counters, unsigned int i,
                             unsigned int n) {
    const unsigned int global_i = get_global_id(0);
    if (global_i * SEGMENT_SIZE < n) {
        unsigned int segment_begin = global_i * SEGMENT_SIZE;
        unsigned int segment_end = (global_i + 1) * SEGMENT_SIZE;
        unsigned int counter = 0;
        for (int j = segment_begin; j < segment_end; j++) {
            counter += (as[j] >> i) & 1;
        }
        counters[global_i] = counter;
    }
}

void local_radix(__local unsigned int *tmp, __local unsigned int *counter, unsigned int offset,
                 __global unsigned int *as) {
    unsigned int tmp_counter = 0;
    unsigned int tmp2[SEGMENT_SIZE];
    for (unsigned int i = 0; i < SEGMENT_SIZE; ++i) {
        tmp2[i] = tmp[i];
        tmp_counter += ((tmp2[i] >> offset) & 1);
    }
    unsigned int i = 0, j = SEGMENT_SIZE - tmp_counter;
    for (unsigned int k = 0; k < SEGMENT_SIZE; ++k) {
        if (!((tmp2[k] >> offset) & 1))
            tmp[i++] = tmp2[k];
        else
            tmp[j++] = tmp2[k];
    }
    *counter = tmp_counter;
}

__kernel void radix_sort(__global const unsigned int *counters, __global unsigned int *as, __global unsigned int *bs,
                         unsigned int offset, unsigned int n) {
    const unsigned int i = get_global_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int local_id = get_local_id(0);
    __local unsigned int counter_sum;
    __local unsigned int counter;
    __local unsigned int tmp[SEGMENT_SIZE];
    if (i < n) {
        unsigned int zero_all_count = n - counters[n / SEGMENT_SIZE - 1];
        tmp[local_id] = as[i];
        barrier(CLK_LOCAL_MEM_FENCE);

        if (i == group_id * SEGMENT_SIZE) {
            if (group_id == 0)
                counter_sum = 0;
            else
                counter_sum = counters[group_id - 1];
            local_radix(tmp, &counter, offset, as);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned int zero_counter = SEGMENT_SIZE - counter;
        unsigned int zero_counter_sum = group_id * SEGMENT_SIZE - counter_sum;
        unsigned int new_pos = (local_id < zero_counter ? zero_counter_sum + local_id
                                                        : zero_all_count + counter_sum + (local_id - zero_counter));
        barrier(CLK_GLOBAL_MEM_FENCE);

        bs[new_pos] = tmp[local_id];
    }
}