#ifndef COUNTER_SIZE
    #define COUNTER_SIZE 4
#endif

#ifndef MASK
    #define MASK (COUNTER_SIZE - 1)
#endif

#ifndef LOCAL_ARRAY_SIZE
    #define LOCAL_ARRAY_SIZE 128
#endif

__kernel void counting(const __global unsigned int *as, __global unsigned int *bs, const unsigned degree,
                       const unsigned size) {
    unsigned gid = get_global_id(0);
    unsigned wid = get_group_id(0);
    unsigned lid = get_local_id(0);
    unsigned groups = get_num_groups(0);

    __local unsigned counter[COUNTER_SIZE];

    if (lid < COUNTER_SIZE) {
        counter[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned index;
    if (gid < size) {
        index = (as[gid] >> degree) & MASK;

        atomic_add(&counter[index], 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < COUNTER_SIZE) {
        bs[groups * lid + wid] = counter[lid];
    }
}

__kernel void prefix_sum(const __global unsigned *as, __global unsigned *bs, const unsigned offset,
                         const unsigned size) {
    unsigned gid = get_global_id(0);

    if (gid >= size) {
        return;
    }

    if (gid >= offset) {
        bs[gid] = as[gid] + as[gid - offset];
    } else {
        bs[gid] = as[gid];
    }
}

__kernel void fill_zero(__global unsigned *as, const unsigned size) {
    unsigned gid = get_global_id(0);

    if (gid < size) {
        as[gid] = 0;
    }
}

__kernel void shift(const __global unsigned *as, __global unsigned *bs, const unsigned size) {
    int gid = get_global_id(0);

    if (gid >= size) {
        return;
    }

    bs[gid] = gid > 0 ? as[gid - 1] : 0;
}

__kernel void radix(__global const unsigned int *as, __global unsigned int *bs, const __global unsigned int *offsets,
                    const unsigned degree, const unsigned size) {
    unsigned gid = get_global_id(0);
    unsigned lid = get_local_id(0);
    unsigned wid = get_group_id(0);
    unsigned gsize = get_local_size(0);
    unsigned groups = get_num_groups(0);

    __local unsigned nums[LOCAL_ARRAY_SIZE];

    nums[lid] = (as[gid] >> degree) & MASK;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid >= size) {
        return;
    }

    unsigned temp, diff = gid / gsize * gsize, count = 0, value = nums[lid];

    for (unsigned i = 0, end = gid - diff, nsize = size - diff; i < end && i < nsize; i++) {
        temp = nums[i];
        count += (temp == value) ? 1 : 0;
    }
    unsigned offset = count;
    bs[offsets[groups * value + wid] + offset] = as[gid];
}
