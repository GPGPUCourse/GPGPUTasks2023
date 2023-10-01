// vim: syntax=c

__kernel void sum_naive(
    __global const unsigned int * a,
    unsigned int n,
    __global unsigned int * result
) {
    const unsigned int i = get_global_id(0);

    if (i < n) {
        atomic_add(result, a[i]);
    }
}

#define VALUES_PER_WORK_ITEM 64

__kernel void sum_loop(
    __global const unsigned int * a,
    unsigned int n,
    __global unsigned int * result
) {
    const unsigned int idx = get_global_id(0);
    const unsigned int l = idx * VALUES_PER_WORK_ITEM;
    const unsigned int r = l + VALUES_PER_WORK_ITEM;

    unsigned int s = 0;
    for (unsigned int i = l; i < r; ++i) {
        if (i >= n) {
            break;
        }

        s += a[i];
    }

    atomic_add(result, s);
}

__kernel void sum_loop_coalesced(
    __global const unsigned int * a,
    unsigned int n,
    __global unsigned int * result
) {
    const unsigned int groupSize = get_local_size(0);
    const unsigned int groupId = get_group_id(0);
    const unsigned int localId = get_local_id(0);

    unsigned int s = 0;
    for (unsigned int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        const unsigned int idx = (groupId * VALUES_PER_WORK_ITEM + i) * groupSize + localId;

        if (idx >= n) {
            continue;
        }

        s += a[idx];
    }

    atomic_add(result, s);
}

#define WORK_GROUP_SIZE 128

__kernel void sum_local(
    __global const unsigned int * a,
    unsigned int n,
    __global unsigned int * result
) {
    const unsigned int globalId = get_global_id(0);
    const unsigned int localId = get_local_id(0);

    __local unsigned int local_xs[WORK_GROUP_SIZE];
    local_xs[localId] = globalId < n ? a[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        unsigned int s = 0;

        for (unsigned int i = 0; i < WORK_GROUP_SIZE; ++i) {
            s += local_xs[i];
        }

        atomic_add(result, s);
    }
}

__kernel void sum_tree(
    __global const unsigned int * a,
    unsigned int n,
    __global unsigned int * result
) {
    const unsigned int globalId = get_global_id(0);
    const unsigned int localId = get_local_id(0);

    __local unsigned int local_xs[WORK_GROUP_SIZE];
    local_xs[localId] = globalId < n ? a[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (unsigned int nvalues = WORK_GROUP_SIZE; nvalues > 1; nvalues /= 2) {
        if (2 * localId < nvalues) {
            local_xs[localId] += local_xs[localId + nvalues / 2];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) {
        atomic_add(result, local_xs[0]);
    }
}
