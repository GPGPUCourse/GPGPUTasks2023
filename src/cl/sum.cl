#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORKITEM 32
#define WORKGROUP_SIZE 128


__kernel void sum_atomic(__global const unsigned int* a,
                         __global       unsigned int* sum,
                         unsigned int n)
{
    unsigned int gid = get_global_id(0);
    if (gid < n)
        atomic_add(sum, a[gid]);
}


__kernel void sum_atomic_loop(__global const unsigned int* a,
                               __global       unsigned int* sum,
                               unsigned int n)
{
    unsigned int gid = get_global_id(0);

    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int idx = gid * VALUES_PER_WORKITEM + i;
        if (idx < n)
            res += a[idx];
    }

    atomic_add(sum, res);
}

__kernel void sum_coalesced_loop(__global const unsigned int* a,
                                  __global       unsigned int* sum,
                                  unsigned int n)
{
    unsigned int lid = get_local_id(0);
    unsigned int wid = get_group_id(0);
    unsigned int grs = get_local_size(0);

    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (idx < n) {
            res += a[idx];
        }
    }

    atomic_add(sum, res);
}

__kernel void sum_local_memory(__global const unsigned int* a,
                               __global       unsigned int* sum,
                               unsigned int n)
{
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];
    
    if (gid < n)
        buf[lid] = a[gid];
    else
        buf[lid] = 0;
    
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int group_res = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i) {
            group_res += buf[i];
        }
        atomic_add(sum, group_res);
    }
}

__kernel void sum_tree(__global const unsigned int *arr,
                           __global unsigned int *sum,
                           unsigned int n)
{
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int wid = get_group_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? arr[gid]: 0;

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
