#line 1

__kernel void sum_gpu_2(__global const unsigned int *arr, __global unsigned int *sum, const unsigned int _) {
    const unsigned int gid = get_global_id(0);
    atomic_add(sum, arr[gid]);
}

#define VALUES_PER_WORKITEM 64
__kernel void sum_gpu_3(__global const unsigned int *arr, __global unsigned int *sum, const unsigned int n) {
    const unsigned int gid = get_global_id(0);
    unsigned int res = 0;

    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        unsigned int idx = gid * VALUES_PER_WORKITEM + i;
        res += arr[idx] * (idx < n);
    }

    atomic_add(sum, res);
}


__kernel void sum_gpu_4(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n) {
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        res += arr[idx] * (idx < n);
    }

    atomic_add(sum, res);
}

#define WORKGROUP_SIZE 128
__kernel void sum_gpu_5(__global const unsigned int *arr, __global unsigned int *sum, unsigned int _) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = arr[gid];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int group_res = 0;
        for (unsigned int i = 0; i < WORKGROUP_SIZE; ++i) {
            group_res += buf[i] * (lid == 0);
        }

        atomic_add(sum, group_res);
    }
}

__kernel void sum_gpu_6(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nVals = WORKGROUP_SIZE; nVals > 1; nVals /= 2) {
        int a = buf[lid];
        int b = buf[nVals / 2 + lid];
        buf[lid] = (a + b) * (2 * lid < nVals);
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buf[0]);
    }
}
