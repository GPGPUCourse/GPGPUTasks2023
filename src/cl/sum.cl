// TODO
__kernel void atomicSum1(__global const uint *arr,
     __global uint *sum, uint n) {
    size_t gid = get_global_id(0);

    if (n <= gid) {
        return;
    }
    atomic_add(sum, arr[gid]);
}


#define VALUES_PER_WORKITEM 32
__kernel void loopSum2(__global const uint *arr,
                      __global uint *sum,
                      uint n) {
    const uint gid = get_global_id(0);

    uint res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = gid * VALUES_PER_WORKITEM + i;
        if (idx < n) {
            res += arr[idx];
        }
    }

    atomic_add(sum, res);
}

__kernel void loopCoalescedSum3(__global const uint *arr,
                              __global uint *sum,
                              uint n) {
    size_t lid = get_local_id(0);
    size_t wid = get_group_id(0);
    size_t grs = get_local_size(0);

    int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; i++) {
        int idx = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (idx < n)
            res += arr[idx];
    }

    atomic_add(sum, res);
}

#define WORKGROUP_SIZE 64
__kernel void localMemSum4(__global const uint *arr,
                          __global uint *sum,
                          uint n) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);

    __local uint buf[WORKGROUP_SIZE];

    buf[lid] = arr[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        uint groupRes = 0;
        for (int i = 0; i < WORKGROUP_SIZE; i++) {
            groupRes += buf[i];
        }
        atomic_add(sum, groupRes);
    }
}

__kernel void treeSum5(__global const uint *arr,
                       __global uint *sum,
                       uint n) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);

    __local uint buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            buf[lid] += buf[lid + nValues / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
       atomic_add(sum, buf[0]);
    };
}
