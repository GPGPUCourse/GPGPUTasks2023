#ifndef COUNTER_SIZE
    #define COUNTER_SIZE 4
#endif

#ifndef MASK
    #define MASK (COUNTER_SIZE - 1)
#endif

#ifndef LOCAL_ARRAY_SIZE
    #define LOCAL_ARRAY_SIZE 128
#endif

__kernel void radix(__global const unsigned int *as, __global unsigned int *bs, const __global unsigned int *offsets,
                    const __global unsigned int *counting, const unsigned degree, const unsigned size) {
    unsigned gid = get_global_id(0);
    unsigned lid = get_local_id(0);
    unsigned wid = get_group_id(0);
    unsigned gsize = get_local_size(0);
    unsigned groups = get_num_groups(0);

    __local unsigned nums[LOCAL_ARRAY_SIZE];


    if (gid < size) {
        nums[lid] = (as[gid] >> degree) & MASK;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid >= size) {
        return;
    }

    unsigned temp, diff = gid / gsize * gsize, count = 0, value = nums[lid];

    for (unsigned i = 0; i < value; i++) {
        count += counting[groups * i + wid];
    }
    unsigned offset = lid - count;
    bs[offsets[groups * value + wid] + offset] = as[gid];
}
