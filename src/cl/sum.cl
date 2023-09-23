__kernel void sum_global_atomic(
    __global const unsigned *array,
    unsigned len,
    volatile __global unsigned *result
) {
    int index = get_global_id(0);
    if (index >= len)
        return;
    atomic_add(result, array[index]);
}

#define VALUES_PER_WORKITEM 128
__kernel void sum_noncoalesced_loop(
    __global const unsigned *array,
    unsigned len,
    volatile __global unsigned *result
) {
    int globalId = get_global_id(0);
    __global const unsigned *from = array + globalId * VALUES_PER_WORKITEM;
    __global const unsigned *to = from + VALUES_PER_WORKITEM;
    if (to > array + len)
        to = array + len;
    unsigned workItemResult = 0;
    for (__global const unsigned *this = from; this < to; ++this)
        workItemResult += *this;
    atomic_add(result, workItemResult);
}
#undef VALUES_PER_WORKITEM

#define VALUES_PER_WORKITEM 128
__kernel void sum_coalesced_loop(
    __global const unsigned *array,
    unsigned len,
    volatile __global unsigned *result
) {
    int groupId = get_group_id(0);
    int localSize = get_local_size(0);
    int localId = get_local_id(0);
    __global const unsigned *from = array + groupId * localSize * VALUES_PER_WORKITEM;
    __global const unsigned *to = from + localSize * VALUES_PER_WORKITEM;
    if (to > array + len)
        to = array + len;
    unsigned workItemResult = 0;
    for (__global const unsigned *this = from + localId; this < to; this += localSize)
        workItemResult += *this;
    atomic_add(result, workItemResult);
}
#undef VALUES_PER_WORKITEM

#define WORK_GROUP_SIZE 1024
__kernel void sum_local_copy(
    __global const unsigned *array,
    unsigned len,
    volatile __global unsigned *result
) {
    __local unsigned local_subarray[WORK_GROUP_SIZE];
    int globalId = get_global_id(0);
    int localId = get_local_id(0);
    local_subarray[localId] = globalId < len ? array[globalId] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (localId != 0)
        return;
    unsigned localResult = 0;
    for (__local unsigned *this = local_subarray; this < local_subarray + WORK_GROUP_SIZE; ++this)
        localResult += *this;
    atomic_add(result, localResult);
}
#undef WORK_GROUP_SIZE

#define WORK_GROUP_SIZE 512
__kernel void sum_tree(
    __global const unsigned *array,
    unsigned len,
    volatile __global unsigned *result
) {
    __local unsigned localBuffer[WORK_GROUP_SIZE];
    int globalId = get_global_id(0);
    int localId = get_local_id(0);
    localBuffer[localId] = globalId < len ? array[globalId] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int localN = WORK_GROUP_SIZE; localN > 1; localN /= 2) {
        if (localId < localN / 2)
            localBuffer[localId] += localBuffer[localId + localN / 2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (localId == 0)
        atomic_add(result, localBuffer[0]);
}
#undef WORK_GROUP_SIZE
