#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define VALUES_PER_WORKITEM 64
#define WORKGROUP_SIZE 128

__kernel void baseline(__global unsigned int *sum, __global const unsigned int *as, const unsigned int n) {
    const unsigned id = get_global_id(0);
    if (id >= n) {
        return;
    }
    atomic_add(sum, as[id]);
}

__kernel void with_cycle(__global unsigned int *sum, __global const unsigned int *as, const unsigned int n) {
    const unsigned id = get_global_id(0);
    
    unsigned int meg_sum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = id * VALUES_PER_WORKITEM + i;
        if (idx < n) {
            meg_sum += as[idx];
        }
    }

    atomic_add(sum, meg_sum);
}

__kernel void coalesced_with_cycle(__global unsigned int *sum, __global const unsigned int *as, const unsigned int n) {
    const unsigned lid = get_local_id(0);
    const unsigned gid = get_group_id(0);
    const unsigned sz  = get_local_size(0);
    
    unsigned int meg_sum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = gid * sz * VALUES_PER_WORKITEM + lid * VALUES_PER_WORKITEM + i;
        if (idx < n) {
            meg_sum += as[idx];
        }
    }

    atomic_add(sum, meg_sum);
}

__kernel void local_memory_and_main_thread(__global unsigned int *sum, __global const unsigned int *as, const unsigned int n) {
    const unsigned lid = get_local_id(0);
    const unsigned gid = get_global_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? as[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    if (lid == 0) {
        unsigned int meg_sum = 0;
        for (int i = 0; i < WORKGROUP_SIZE; ++i) {
            meg_sum += buf[i];
        }

        atomic_add(sum, meg_sum);
    }
}

__kernel void with_tree(__global unsigned int *sum, __global const unsigned int *as, const unsigned int n) {
    const unsigned lid = get_local_id(0);
    const unsigned gid = get_global_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? as[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int m = WORKGROUP_SIZE; m > 1; m >>= 1) {
        if (2 * lid < m) {
            unsigned int a = buf[lid];
            unsigned int b = buf[lid + m/2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buf[0]);
    }
}