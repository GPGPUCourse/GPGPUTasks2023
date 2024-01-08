#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORKITEM_VALUES 128
#define WORKGROUP_SIZE 256

__kernel void sum_global_atomic_add(__global const unsigned *arr,
                                    int len,
                                    __global unsigned* res)
{
    const unsigned index = get_global_id(0);

    if (index >= len)
        return;

    atomic_add(res, arr[index]);
}

__kernel void sum_noncoalesced_loop(__global const unsigned int* arr,
                                    int len,
                                    __global unsigned* res)
{
    unsigned sum = 0;

    for (int i = 0; i < WORKITEM_VALUES; ++i) {
        int index = get_global_id(0) * WORKITEM_VALUES + i;
        if (index >= len)
            break;
        sum += arr[index];
    }
    atomic_add(res, sum);
}

__kernel void sum_coalesced_loop(__global const unsigned int* arr,
                                 int len,
                                 __global unsigned* res)
{
    unsigned sum = 0;
    unsigned localId = get_local_id(0);
    unsigned workgroupId = get_group_id(0);
    unsigned localSize = get_local_size(0);

    for (int i = 0; i < WORKITEM_VALUES; ++i) {
        int index = workgroupId * localSize * WORKITEM_VALUES + localId + i * localSize;
        if (index >= len)
            break;
        sum += arr[index];
    }
    atomic_add(res, sum);
}

__kernel void sum_local(__global const unsigned int* arr,
                        int len,
                        __global unsigned* res)
{
    unsigned localId = get_local_id(0);
    unsigned globalId = get_global_id(0);
    unsigned group_res = 0;
    __local unsigned buffer[WORKGROUP_SIZE];

    if (globalId < len)
        buffer[localId] = arr[globalId];
    else
        buffer[localId] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        for (int i = 0; i < WORKGROUP_SIZE; ++i)
            group_res += buffer[i];
        atomic_add(res, group_res);
    }
}

__kernel void sum_tree(__global const unsigned int* arr,
                        int len,
                        __global unsigned* res)
{
    unsigned group_res = 0;
    unsigned globalId = get_global_id(0);
    unsigned localId = get_local_id(0);
    __local unsigned buffer[WORKGROUP_SIZE];

    if (globalId < len)
        buffer[localId] = arr[globalId];
    else
        buffer[localId] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    int numVals = WORKGROUP_SIZE;
    for (; numVals > 1; numVals >>= 1) {
        if (2 * localId < numVals)
            buffer[localId] += buffer[localId + (numVals >> 1)];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (localId == 0)
        atomic_add(res, buffer[0]);
}