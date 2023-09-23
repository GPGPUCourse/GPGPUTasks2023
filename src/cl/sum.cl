#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6


__kernel void sum_global_atomic(__global unsigned int *sum, __global const unsigned int *as, unsigned int n) {
    const unsigned int i = get_global_id(0);

    atomic_add(sum, as[i]);
}

#define VALUES_PER_WORKITEM 64
__kernel void sum_for(__global unsigned int *sum, __global const unsigned int *as, unsigned int n) {
    const unsigned int global_id = get_global_id(0);

    unsigned int local_sum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int as_i = global_id * VALUES_PER_WORKITEM + i;
        if (as_i < n) {
            local_sum += as[as_i];
        }
    }

    atomic_add(sum, local_sum);
}

#define VALUES_PER_WORKITEM 64
__kernel void sum_for_coalesed(__global unsigned int *sum, __global const unsigned int *as, unsigned int n) {
    const unsigned int local_id = get_local_id(0);
    const unsigned int workgroup_id = get_group_id(0);
    const unsigned int group_size = get_local_size(0);

    unsigned int local_sum = 0;
    for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
        int as_i = workgroup_id * group_size * VALUES_PER_WORKITEM + i * group_size + local_id;
        if (as_i < n) {
            local_sum += as[as_i];
        }
    }

    atomic_add(sum, local_sum);
}

#define WORKGROUP_SIZE 128
__kernel void sum_local_memory(__global unsigned int *sum, __global const unsigned int *as, unsigned int n) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];
    if (global_id < n) {
        buf[local_id] = as[global_id];
    } else {
        buf[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        unsigned int group_sum = 0;
        for (int i = 0; i < WORKGROUP_SIZE; ++i) {
            group_sum += buf[i];
        }
        atomic_add(sum, group_sum);
    }
}

#define WORKGROUP_SIZE 128
__kernel void sum_tree(__global unsigned int *sum, __global const unsigned int *as, unsigned int n) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int work_group_id = get_group_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];
    buf[local_id] = global_id < n ? as[global_id] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2  * local_id < nValues) {
            unsigned int a = buf[local_id];
            unsigned int b = buf[local_id + nValues / 2];
            buf[local_id] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_id == 0) {
        atomic_add(sum, buf[0]);
    }
}
