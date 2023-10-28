#ifndef COUNTER_SIZE
    #define COUNTER_SIZE 4
#endif

#ifndef MASK
    #define MASK (COUNTER_SIZE - 1)
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

__kernel void counting_without_transpose(const __global unsigned int *as, __global unsigned int *bs,
                                         const unsigned degree, const unsigned size) {
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
        bs[COUNTER_SIZE * wid + lid] = counter[lid];
    }
}
