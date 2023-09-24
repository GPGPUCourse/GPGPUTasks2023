#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void baseline(__global const unsigned int *a, __global unsigned int *result, unsigned int n) {
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    atomic_add(result, a[index]);
}

__kernel void cycle_n(__global const unsigned int *a, __global unsigned int *result, unsigned int n,
                      unsigned int values_per_workitem) {
    const unsigned global_index = get_global_id(0);

    unsigned int sum = 0;
    for (unsigned int i = 0; i < values_per_workitem; i++) {
        unsigned int index = global_index * values_per_workitem + i;
        if (index < n)
            sum += a[index];
    }

    atomic_add(result, sum);
}

__kernel void cycle_3(__global const unsigned int *a, __global unsigned int *result, unsigned int n) {
    cycle_n(a, result, n, 3);
}

__kernel void cycle_64(__global const unsigned int *a, __global unsigned int *result, unsigned int n) {
    cycle_n(a, result, n, 64);
}

__kernel void cycle_coalesced_n(__global const unsigned int *a, __global unsigned int *result, unsigned int n,
                                unsigned int values_per_workitem) {
    const unsigned int local_index = get_local_id(0);
    const unsigned int group_index = get_group_id(0);
    const unsigned int local_size = get_local_size(0);

    unsigned int sum = 0;

    for (unsigned int i = 0; i < values_per_workitem; i++) {
        unsigned int index = group_index * local_size * values_per_workitem + i * local_size + local_index;
        if (index < n) {
            sum += a[index];
        }
    }

    atomic_add(result, sum);
}

__kernel void cycle_coalesced_4(__global const unsigned int *a, __global unsigned int *result, unsigned int n) {
    cycle_coalesced_n(a, result, n, 4);
}

__kernel void cycle_coalesced_64(__global const unsigned int *a, __global unsigned int *result, unsigned int n) {
    cycle_coalesced_n(a, result, n, 64);
}

#define WORKGROUP_SIZE 128
__kernel void local_mem(__global const unsigned int *a, __global unsigned int *result, unsigned int n) {
    const unsigned int global_index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);

    __local unsigned int buffer[WORKGROUP_SIZE];

    buffer[local_index] = global_index < n ? a[global_index] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_index == 0) {
        unsigned int sum = 0;
        for (int i = 0; i < WORKGROUP_SIZE; i++) {
            sum += buffer[i];
        }
        atomic_add(result, sum);
    }
}

__kernel void tree(__global const unsigned int *a, __global unsigned int *result, unsigned int n) {
    const unsigned int global_index = get_global_id(0);
    const unsigned int local_index = get_local_id(0);

    __local unsigned int buffer[WORKGROUP_SIZE];

    buffer[local_index] = global_index < n ? a[global_index] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues /= 2) {
        if (2 * local_index < nValues) {
            buffer[local_index] += buffer[local_index + nValues / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (local_index == 0) {
        atomic_add(result, buffer[0]);
    }
}
