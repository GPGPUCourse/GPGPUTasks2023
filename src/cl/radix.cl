#define DIGITS_NUMBER 4

__kernel void radix_count(
    __global unsigned int *as, 
    const unsigned int n,
     __global unsigned int *counters,
    const unsigned int working_groups_number,
     const unsigned int mask_offset
) {
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const unsigned int local_i = get_local_id(0);
    const unsigned int mask = DIGITS_NUMBER - 1;
    __local unsigned int local_counters[DIGITS_NUMBER];
    // if (i < 64)
    //     printf("%d, %d, %d, %d\n", i, mask_offset, as[i], (as[i] >> mask_offset) & mask);
    if (local_i < DIGITS_NUMBER) {
        local_counters[local_i] = 0;
    }
    atomic_inc(&local_counters[(as[i] >> mask_offset) & mask]);
    if (local_i < DIGITS_NUMBER) {
        counters[local_i * working_groups_number + get_group_id(0)] = local_counters[local_i];
    }
}

__kernel void radix_prefix_sum(
    const __global unsigned int *counters,
     __global unsigned int *prefix_sums_dst,
    const unsigned int working_groups_number,
    const unsigned int cur_block_size
) {
    const unsigned int i = get_global_id(0);
    if (i >= working_groups_number) {
        return;
    }
    if (((i + 1) & cur_block_size) != 0) {
        unsigned int offset = i;
        unsigned int mask = ~0u;
        while (offset + cur_block_size > i + 1) {
            offset &= mask;
            mask <<= 1;
        }
        if (cur_block_size + offset > 0) {
            for (unsigned int cur_bit_offset = 0; cur_bit_offset < DIGITS_NUMBER * working_groups_number;
                 cur_bit_offset += working_groups_number) {
                prefix_sums_dst[cur_bit_offset + i] += counters[cur_bit_offset + cur_block_size + offset - 1];
            }
        }
    }
}

__kernel void radix_prefix_sum_reduce(
    __global unsigned int *counters,
    const unsigned int working_groups_number,
    const unsigned int cur_block_size
) {
    const unsigned int i = get_global_id(0);
    if ((i + 1) * cur_block_size < working_groups_number + 1 && i * cur_block_size > 0) {
        for (unsigned int cur_bit_offset = 0; cur_bit_offset < DIGITS_NUMBER * working_groups_number;
             cur_bit_offset += working_groups_number) {
            counters[cur_bit_offset + (i + 1) * cur_block_size - 1] += counters[cur_bit_offset + i * cur_block_size - 1];
        }
    }
}

__kernel void radix_sort(
    const __global unsigned int *as,
    __global unsigned int *as_sorted_dst,
    const unsigned int n,
    __global unsigned int *prefix_sums,
    const unsigned int working_groups_number,
    const unsigned int mask_offset
) {
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    const unsigned int mask = DIGITS_NUMBER - 1;
    const unsigned int bits_value = (as[i] >> mask_offset) & mask;
    unsigned int dst_i = 0;
    for (unsigned int p = 0; p < bits_value; ++p) {
        dst_i += prefix_sums[(p + 1) * working_groups_number - 1];
    }
    if (get_group_id(0) > 0) {
        dst_i += prefix_sums[bits_value * working_groups_number + get_group_id(0) - 1];
    }
    const unsigned int wg_size = get_local_size(0);
    for (unsigned int p = i - (i % wg_size); p < i; ++p) {
        if (((as[p] >> mask_offset) & mask) == bits_value) {
            dst_i++;
        }
    }
    as_sorted_dst[dst_i] = as[i];
}



