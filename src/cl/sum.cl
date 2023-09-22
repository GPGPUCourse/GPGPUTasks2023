#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum(__global unsigned int *a, __global unsigned int *c, unsigned int n) {
    const unsigned int index = get_global_id(0);
    if (index >= n)
        return;
    atomic_add(c, a[index]);
}

#define VALUES_PER_WORKITEM 32
__kernel void loop(__global unsigned int *a, __global unsigned int *c, unsigned int n) {
    const unsigned int index = get_global_id(0);
    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = index * VALUES_PER_WORKITEM + i;
        if (idx < n)
            sum += a[idx];
    }
    atomic_add(c, sum);
}

__kernel void loop_coalesced(__global unsigned int *a, __global unsigned int *c, unsigned int n) {
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int grs = get_local_size(0);
    unsigned int sum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int idx = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
        if (idx < n) {
            sum += a[idx];
        }
    }
    atomic_add(c, sum);
}

#define WORKGROUP_SIZE 128
__kernel void sum_5(__global unsigned int *a, __global unsigned int *c, unsigned int n) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    __local unsigned int buf[WORKGROUP_SIZE];
    buf[lid] = a[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < WORKGROUP_SIZE; ++i) {
            sum += buf[i];
        }
        atomic_add(c, sum);
    }
}

__kernel void sum_6(__global unsigned int *a, __global unsigned int *c, unsigned int n) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    __local unsigned int sum[WORKGROUP_SIZE];
    sum[lid] = gid < n ? a[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            unsigned int a = sum[lid];
            unsigned int b = sum[lid + nValues / 2];
            sum[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        atomic_add(c, sum[0]);
    }
}

__kernel void sum_7(__global unsigned int *a, __global unsigned int *c, unsigned int n) {
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    __local unsigned int sum[WORKGROUP_SIZE];
    sum[lid] = gid < n ? a[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * lid < nValues) {
            unsigned int a = sum[lid];
            unsigned int b = sum[lid + nValues / 2];
            sum[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        c[wid] = sum[0];
    }
}
