#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define VALUES_PER_WORK_ITEM 32
#define WORKGROUP_SIZE 128

__kernel void globalAtomSum(__global const int *array, const unsigned int array_size, __global unsigned int *sum) {
    unsigned int idx = get_global_id(0);
    if (idx < array_size) {
        atomic_add(sum, array[idx]);
    }
}

__kernel void loopSum(__global const int *array, const unsigned int array_size, __global unsigned int *sum) {
    const unsigned int idx = get_global_id(0);
    unsigned int res = 0;
    for (int i = idx * VALUES_PER_WORK_ITEM; i < (idx + 1) * VALUES_PER_WORK_ITEM; ++i) {
        if (i < array_size) {
            res += array[i];
        }
    }

    atomic_add(sum, res);
}

__kernel void loopCoalescedSum(__global const int *array, const unsigned int array_size, __global unsigned int *sum) {
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);
    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        int idx = wid * grs * VALUES_PER_WORK_ITEM + i * grs + lid;
        if (idx < array_size) {
            res += array[idx];
        }
    }

    atomic_add(sum, res);
}

__kernel void sumWithLocalMemes(__global const int *array, const unsigned int array_size, __global unsigned int *sum) {
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);

    __local unsigned int buf [WORKGROUP_SIZE];
    buf[lid] = gid < array_size ? array[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0)
    {
        unsigned int group_res = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i)
        {
            group_res += buf[i];
        }

        atomic_add(sum, group_res);
    }
}

__kernel void treeSum(__global const int *array, const unsigned int array_size, __global unsigned int *sum) {
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);

    __local unsigned int buf [WORKGROUP_SIZE];
    buf[lid] = gid < array_size ? array[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2)
    {
        if (2 * lid < nValues)
        {
            unsigned int a = buf[lid];
            unsigned int b = buf[lid + nValues / 2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        atomic_add(sum, buf[0]);
    }
}
