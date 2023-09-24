#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORKITEM 128
#define WORKGROUP_SIZE 128

__kernel void only_atomic(__global unsigned int *result,
                     __global const unsigned int* a,
                     unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    atomic_add(result, a[index]);
}

__kernel void cycle(__global unsigned int *result,
                 __global const unsigned int* a,
                 unsigned int n)
{
    const unsigned int gid = get_global_id(0);

    unsigned int res = 0;

    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int index = gid * VALUES_PER_WORKITEM + i;
        if (index < n)
            res += a[index];
    }

    atomic_add(result, res);
}

__kernel void coalesced(__global unsigned int *result,
                 __global const unsigned int* a,
                 unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int res = 0;

    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int index = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (index < n)
            res += a[index];
    }

    atomic_add(result, res);
}

__kernel void mem_local(__global unsigned int *result,
                 __global const unsigned int* a,
                 unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = a[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int group_res = 0;
        for (int i = 0; i < WORKGROUP_SIZE; i++) {
            group_res += buf[i];
        }
        atomic_add(result, group_res);
    }
}

__kernel void tree(__global unsigned int *result,
                 __global const unsigned int* a,
                 unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? a[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nv = WORKGROUP_SIZE; nv > 1; nv /= 2) {
        if (2 * lid < nv) {
            unsigned int a = buf[lid];
            unsigned int b = buf[lid + nv / 2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(result, buf[lid]);
    }
}
