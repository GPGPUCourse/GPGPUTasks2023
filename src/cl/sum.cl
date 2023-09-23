#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum_baseline(__global const unsigned int* src,
                           __global unsigned int* dest,
                           unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    atomic_add(dest, src[index]);
}

// keep synced with same variables in main_sum.cpp
#define VALUES_PER_WORKITEM 64
#define WORKGROUP_SIZE 64

__kernel void sum_loop(__global const unsigned int* src,
                       __global unsigned int* dest,
                       unsigned int n)
{
    const unsigned int gid = get_global_id(0);

    unsigned int local_res = 0;
    for (unsigned int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        unsigned int index = i + gid * VALUES_PER_WORKITEM;

        if (index >= n) break;

        local_res += src[index];
    }

    atomic_add(dest, local_res);
}

__kernel void sum_loop_coalesced(__global const unsigned int* src,
                                 __global unsigned int* dest,
                                 unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int local_res = 0;
    const unsigned int offset = wid * grs * VALUES_PER_WORKITEM;

    for (unsigned int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        const unsigned int index = lid + i * grs + offset;

        if (index >= n) break;

        local_res += src[index];
    }

    atomic_add(dest, local_res);
}


__kernel void sum_local_mem(__global const unsigned int* src,
                            __global unsigned int* dest,
                            unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);

    if (gid > n) return;

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = src[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid != 0) return;

    unsigned int local_res = 0;
    for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i) {
        local_res += buf[i];
    }

    atomic_add(dest, local_res);
}

__kernel void sum_tree(__global const unsigned int* src,
                       __global unsigned int* dest,
                       unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int gid = get_global_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? src[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int local_res = 0;
    for (unsigned int alive = WORKGROUP_SIZE; alive > 1; alive >>= 1) {
        unsigned int L = lid;
        unsigned int R = alive / 2 + lid;
        if (R < alive) {
            buf[L] += buf[R];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) dest[wid] = buf[0];
}
