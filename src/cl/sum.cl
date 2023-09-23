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

__kernel void BatchSum(__global const int *array, const unsigned int array_size, __global unsigned int *sum) {
    const unsigned int gid = get_global_id(0);
    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        unsigned int idx = gid * VALUES_PER_WORK_ITEM + i;
        if (idx >= array_size) {
            atomic_add(sum, res);
            return;
        }
        res += array[idx];
    }
    atomic_add(sum, res);
}

__kernel void BatchSumCoalesed(__global const int *array, const unsigned int array_size, __global unsigned int *sum) {
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORK_ITEM; ++i) {
        unsigned int idx = wid * grs * VALUES_PER_WORK_ITEM + i * grs + lid;
        if (idx >= array_size) {
            atomic_add(sum, res);
            return;
        }
        res += array[idx];
    }
    atomic_add(sum, res);
}

__kernel void LocalMemSUm(__global const int *array, const unsigned int array_size, __global unsigned int *sum) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buffer[WORKGROUP_SIZE];

    buffer[lid] = array[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        unsigned int group_res = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i) {
            group_res += buffer[i];
        }
        atomic_add(sum, group_res);
    }
}

__kernel void TreeSum(__global const int *array, const unsigned int array_size, __global unsigned int *sum) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buffer[WORKGROUP_SIZE];

    buffer[lid] = gid < array_size ? array[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int n_values = WORKGROUP_SIZE; n_values > 1; n_values /= 2) {
        if (2 * lid < n_values) {
            unsigned int a = buffer[lid];
            unsigned int b = buffer[lid + n_values / 2];
            buffer[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buffer[0]);
    }
}