#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

// Same as in main.
#define WORK_GROUP_SIZE 128
#define VALUES_PER_WORKITEM 128

// Baseline.
__kernel void baseline(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n) {
    const unsigned gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    atomic_add(sum, arr[gid]);
}


// Cycle with non coalesced access.
__kernel void cycle(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n) {
    const unsigned gid = get_global_id(0);

    unsigned res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = gid * VALUES_PER_WORKITEM + i;
        if (idx < n) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}

// Cycle with coalesced access.
__kernel void cycle_coalesced(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n) {
    const unsigned lid = get_local_id(0);
    const unsigned wid = get_group_id(0);
    const unsigned grs = get_local_size(0);

    unsigned res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (idx < n) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}

// Local memory.
__kernel void local_mem(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n) {
    const unsigned gid = get_global_id(0);
    const unsigned lid = get_local_id(0);

    __local unsigned buf[WORK_GROUP_SIZE];
    buf[lid] = arr[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned group_res = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            group_res += buf[i];
        }
        atomic_add(sum, group_res);
    }
}

// Tree.
__kernel void tree(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n) {
    const unsigned gid = get_global_id(0);
    const unsigned lid = get_local_id(0);

    __local unsigned buf[WORK_GROUP_SIZE];
    buf[lid] = gid < n ? arr[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORK_GROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            unsigned a = buf[lid];
            unsigned b = buf[lid + nValues / 2];
            buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buf[0]);
    }
}
