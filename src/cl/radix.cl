#ifndef ARRAY_SIZE
    #define ARRAY_SIZE 4
#endif

#ifndef LOCAL_ARRAY_SIZE
    #define LOCAL_ARRAY_SIZE 128
#endif

__kernel void counting(__global unsigned int *as, __global unsigned int *bs, const unsigned degree, unsigned size) {
    unsigned gid = get_global_id(0);
    unsigned wid = get_group_id(0);
    unsigned lid = get_local_id(0);
    unsigned groups = get_num_groups(0);

    __local unsigned counter[ARRAY_SIZE];

    if (lid < 4) {
        counter[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned v;
    if (gid < size) {
        v = (as[gid] >> degree) & 3;

        atomic_add(&counter[v], 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < size) {
        bs[groups * v + wid] = counter[v];
    }
}

__kernel void prefix_sum(__global unsigned *as, __global unsigned *bs, const unsigned offset, const unsigned size) {
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

__kernel void shift(__global const unsigned *as, __global unsigned *bs, const unsigned size) {
    int gid = get_global_id(0);

    if (gid >= size) {
        return;
    }

    bs[gid] = gid > 0 ? as[gid - 1] : 0;
}

__kernel void radix(__global const unsigned int *as, __global unsigned int *bs, __global const unsigned int *offsets,
                    const unsigned degree, const unsigned size) {
    unsigned gid = get_global_id(0);
    unsigned lid = get_local_id(0);
    unsigned wid = get_group_id(0);
    unsigned gsize = get_local_size(0);
    unsigned groups = get_num_groups(0);

    __local unsigned nums[LOCAL_ARRAY_SIZE];

    nums[lid] = (as[gid] >> degree) & 3;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid >= size) {
        return;
    }

    unsigned v, diff = gid / gsize * gsize, count = 0, myv = nums[lid];

    for (unsigned i = 0, end = gid - diff, nsize = size - diff; i < end && i < nsize; i++) {
        v = nums[i];
        count += (v == myv) ? 1 : 0;
    }
    unsigned offset = count;
    bs[offsets[groups * myv + wid] + offset] = as[gid];
}
