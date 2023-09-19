#define VALUES_PER_ITEM 32
#define WORK_GROUP_SIZE 128

__kernel void sumBase(__global unsigned int* arr, __global unsigned int* sum, unsigned int n) {
    const unsigned int idx = get_global_id(0);

    if (idx < n) {
        atomic_add(sum, arr[idx]);
    }
}


__kernel void sumLoop(__global unsigned int* arr, __global unsigned int* sum, unsigned int n) {
    const unsigned int idx = get_global_id(0);

    unsigned int res = 0;
    for (int i = idx * VALUES_PER_ITEM; i < (idx + 1) * VALUES_PER_ITEM; ++i) {
        if (i < n) {
            res += arr[i];
        }
    }
    atomic_add(sum, res);
}


__kernel void sumLoopCoalesced(__global unsigned int* arr, __global unsigned int* sum, unsigned int n) {
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);

    unsigned int res = 0;
    for (int i = 0; i < VALUES_PER_ITEM; ++i) {
        unsigned int idx = wid * grs * VALUES_PER_ITEM + i * grs + lid;
        if (idx < n) {
            res += arr[idx];
        }
    }
    atomic_add(sum, res);
}

__kernel void sumWithMainThread(__global unsigned int* arr, __global unsigned int* sum, unsigned int n) {
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);

    __local unsigned int buf[WORK_GROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int group_res = 0;
        for (int i = 0; i < WORK_GROUP_SIZE; ++i) {
            group_res += buf[i];
        }
        atomic_add(sum, group_res);
    }
}

__kernel void sumTree(__global unsigned int* arr, __global unsigned int* sum, unsigned int n) {
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);
    const unsigned int wid = get_group_id(0);

    __local unsigned int buf[WORK_GROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORK_GROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            buf[lid] += buf[lid + nValues / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        atomic_add(sum, buf[0]);
    }
}

__kernel void sumTree2(__global unsigned int* arr, __global unsigned int* sum, unsigned int n) {
    const unsigned int lid = get_local_id(0);
    const unsigned int gid = get_global_id(0);
    const unsigned int wid = get_group_id(0);

    __local unsigned int buf[WORK_GROUP_SIZE];

    buf[lid] = gid < n ? arr[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORK_GROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            buf[lid] += buf[lid + nValues / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        sum[wid] = buf[0];
    }
}
