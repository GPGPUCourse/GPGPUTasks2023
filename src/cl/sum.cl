__kernel void sum_global_atomic(__global const unsigned int *arr,
                                unsigned int len,
                                __global unsigned int *res)
{
    const unsigned int idx = get_global_id(0);
    const unsigned int val = idx >= len ? 0 : arr[idx];
    atomic_add(res, val);
}

__kernel void sum_loop(__global const unsigned int *arr,
                       unsigned int len,
                       __global unsigned int *sum)
{
    const unsigned int gid = get_global_id(0);

    int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int idx = gid * (int)VALUES_PER_WORKITEM + i;
        if (idx < len) {
            res += arr[idx];
        }
    }
    atomic_add(sum, res);
}

__kernel void sum_loop_coalesced(__global const unsigned int *arr,
                               unsigned int len,
                               __global unsigned int *sum)
{
    const unsigned int localId = get_local_id(0);
    const unsigned int groupId = get_group_id(0);
    const unsigned int groupSize = get_local_size(0);

    const unsigned int offset = groupSize * VALUES_PER_WORKITEM * groupId + localId;

    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int idx = offset + groupSize * i;
        if (idx < len)
            res += arr[idx];
    }
    atomic_add(sum, res);
}

__kernel void sum_local_mem(__global const unsigned int *arr,
                          unsigned int len,
                          __global unsigned int *sum)
{
    const unsigned int globalId = get_global_id(0);
    const unsigned int localId = get_local_id(0);

    __local unsigned int mem[WORKGROUP_SIZE];

    mem[localId] = globalId < len ? arr[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        unsigned int res = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; i++) {
            res += mem[i];
        }
        atomic_add(sum, res);
    }
}

__kernel void sum_local_mem_and_tree(__global const unsigned int *arr,
                                      unsigned int len,
                                      __global unsigned int *sum)
{
    const unsigned int globalId = get_global_id(0);
    const unsigned int localId = get_local_id(0);

    __local unsigned int mem[WORKGROUP_SIZE];

    mem[localId] = globalId < len ? arr[globalId] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int memLen = WORKGROUP_SIZE; memLen > 1; memLen /= 2) {
        if (2 * localId < memLen) {
            mem[localId] += mem[localId + memLen / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0) {
        atomic_add(sum, mem[0]);
    }
}