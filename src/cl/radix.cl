// vim: filetype=c

#define NBITS 4
#define NVALS (1 << NBITS)

__kernel void radix_count(
    __global const unsigned int * as,
    __global unsigned int * cnt,
    unsigned int iter
) {
    __local unsigned int local_cnt[NVALS];

    const unsigned int gi = get_global_id(0);
    const unsigned int li = get_local_id(0);
    const unsigned int wgi = get_group_id(0);

    if (li < NVALS) {
        local_cnt[li] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_inc(&local_cnt[(as[gi] >> (iter * NBITS)) & (NVALS - 1)]);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (li < NVALS) {
        cnt[wgi * NVALS + li] = local_cnt[li];
    }
}

__kernel void radix_sort(
    __global const unsigned int * as,
    __global const unsigned int * cnt,
    __global const unsigned int * cnt_prefix,
    __global unsigned int * bs,
    unsigned int iter
) {
    __local unsigned int local_cnt[NVALS];
    __local unsigned int local_cnt_prefix[NVALS];

    const unsigned int gi = get_global_id(0);
    const unsigned int li = get_local_id(0);
    const unsigned int wgi = get_group_id(0);
    const unsigned int wgc = get_num_groups(0);
    const unsigned int wgs = get_local_size(0);

    if (li < NVALS) {
        local_cnt[li] = cnt[wgi * NVALS + li];
        local_cnt_prefix[li] = cnt_prefix[li * wgc + wgi];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const unsigned int my_value = as[gi];
    const unsigned int my_digit = (my_value >> (iter * NBITS)) & (NVALS - 1);
    unsigned int offset = local_cnt_prefix[my_digit] - local_cnt[my_digit];

    for (unsigned int i = 0; i < li; ++i) {
        const unsigned int d = (as[wgi * wgs + i] >> (iter * NBITS)) & (NVALS - 1);

        if (d == my_digit) {
            ++offset;
        }
    }

    bs[offset] = my_value;
}
