
__kernel void atomicAddSum(__global const unsigned int *a, __global unsigned int *sum, unsigned int n) {
    size_t gid = get_global_id(0);
    atomic_add(sum, a[gid]);
}

#ifndef VALUES_PER_WORKITEM
    #define VALUES_PER_WORKITEM 32
#endif
__kernel void loopSum(__global const unsigned int *a, __global unsigned int *sum, unsigned int n) {
    size_t gid = get_global_id(0);
    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int idx = VALUES_PER_WORKITEM * gid + i;
        if (idx < n)
            res += a[idx];
    }

    atomic_add(sum, res);
}

__kernel void loopCoalesedSum(__global const unsigned int *a, __global unsigned int *sum, unsigned int n) {
    size_t lid = get_local_id(0);
    size_t gr_id = get_group_id(0);
    size_t grs = get_local_size(0);
    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int idx = gr_id * grs * VALUES_PER_WORKITEM + grs * i + lid;
        if (idx < n)
            res += a[idx];
    }

    atomic_add(sum, res);
}


#define WORKGROUP_SIZE 64
__kernel void localMemSum(__global const unsigned int *a, __global unsigned int *sum, unsigned int n) {
    __local unsigned int buffer[WORKGROUP_SIZE];
    size_t lid = get_local_id(0);
    size_t gid = get_global_id(0);
    buffer[lid] = a[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int res = 0;
        for (int i = 0; i < WORKGROUP_SIZE; i++) {
            res += buffer[i];
        }
        atomic_add(sum, res);
    }
}

__kernel void treeSum(__global const unsigned int *a, __global unsigned int *sum, unsigned int n) {
    __local unsigned int buffer[WORKGROUP_SIZE];
    size_t lid = get_local_id(0);
    size_t gid = get_global_id(0);

    buffer[lid] = a[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (lid < nValues / 2) {
            buffer[lid] += buffer[lid + nValues / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        atomic_add(sum, buffer[0]);
    }
}