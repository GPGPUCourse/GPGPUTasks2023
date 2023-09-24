#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define VALUES_PER_WORKITEM 8
#define WORKGROUP_SIZE 128

__kernel void sum_baseline(__global const unsigned int* arr,
                           __global unsigned int* sum,
                           unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    atomic_add(sum, arr[gid]);
}

__kernel void sum_looped(__global const unsigned int* arr,
                           __global unsigned int* sum,
                           unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    unsigned int res = 0;
    for (unsigned int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = gid * VALUES_PER_WORKITEM + i;
        if (idx < n) {
            res += arr[idx];
        }
    }
    atomic_add(sum, res);
}

__kernel void sum_looped_coalesced(__global const unsigned int* arr,
                         __global unsigned int* sum,
                         unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int res = 0;
    for (unsigned int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (idx < n) {
            res += arr[idx];
        }
    }
    atomic_add(sum, res);
}

__kernel void sum_with_local(__global const unsigned int* arr,
                                   __global unsigned int* sum,
                                   unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];
    buf[lid] = gid < n ? arr[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid > 0) {
        return;
    }
    unsigned int res = 0;
    for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i) {
        res += buf[i];
    }
    atomic_add(sum, res);
}

__kernel void sum_tree(__global const unsigned int* arr,
                             __global unsigned int* sum,
                             unsigned int n)
{
    const unsigned int wid = get_group_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];
    buf[lid] = gid < n ? arr[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            unsigned int a = buf[lid];
            unsigned int b = buf[lid + nValues/2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buf[0]);
    }
}

__kernel void sum_tree2(__global const unsigned int* arr,
                       __global unsigned int* sum,
                       unsigned int n)
{
    const unsigned int wid = get_group_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];
    buf[lid] = gid < n ? arr[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            unsigned int a = buf[lid];
            unsigned int b = buf[lid + nValues/2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        sum[wid] = buf[0];
    }
}