#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORKITEM 32
#define WORKGROUP_SIZE 128

__kernel void sum1(__global const uint* array,
                   __global uint* sum,
                   uint n) {
    const size_t gid = get_global_id(0);
    atomic_add(sum, array[gid]);
}

__kernel void sum2(__global const uint* arr,
                        __global uint* sum,
                        uint n) {
    const uint gid = get_global_id(0);

    int result = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int index = gid * VALUES_PER_WORKITEM + i;
        if (index < n) {
            result += arr[index];
        }
    }

    atomic_add(sum, result);
}

__kernel void sum3(__global const uint *arr,
                        __global uint *sum,
                        uint n) {
    const uint lid = get_local_id(0);
    const uint wid = get_group_id(0);
    const uint grs = get_local_size(0);

    int result = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int index = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (index < n)
            result += arr[index];
    }

    atomic_add(sum, result);
}


__kernel void sum4(__global const uint* array,
                   __global uint* sum,
                   uint n) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buffer[WORKGROUP_SIZE];

    buffer[lid] = array[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        uint groupResult = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i)
            groupResult += buffer[i];
        atomic_add(sum, groupResult);
    }
}

__kernel void sum5(__global const uint *arr,
                   __global uint *sum,
                   uint n) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint wid = get_group_id(0);

    __local unsigned int buffer[WORKGROUP_SIZE];
    buffer[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            unsigned int a = buffer[lid];
            unsigned int b = buffer[lid + nValues/2];
            buffer[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
        atomic_add(sum, buffer[0]);
}
