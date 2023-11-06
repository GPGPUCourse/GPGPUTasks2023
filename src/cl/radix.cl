#ifdef __Clocal_yON_IDE__
    #include <local_ybgpu/opencl/cl/clocal_yon_defines.cl>
#endif

#line 5

#define BITS 2
#define VALUES (1 << BITS)
#define value(n, block_id) ((n >> (block_id * BITS)) & (VALUES - 1))

__kernel void counter(__global const unsigned int *as, __global unsigned int *counter, unsigned int block_id) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int local_id = get_local_id(0);

    __local unsigned int local_counter[VALUES];

    if (local_id < VALUES)
        local_counter[local_id] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_inc(&local_counter[value(as[global_id], block_id)]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < VALUES)
        counter[group_id * VALUES + local_id] = local_counter[local_id];
}

#define WORKGROUP_SIZE_X 16
#define WORKGROUP_SIZE_Y 16
#define WORKGROUP_SIZE (WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y)
__kernel void matrix_transpose(__global const float *as, __global float *as_t, unsigned int K, unsigned int M) {
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);

    const unsigned int group_x = get_group_id(0);
    const unsigned int group_y = get_group_id(1);

    __local float buffer[WORKGROUP_SIZE];
    unsigned int local_id = local_y * WORKGROUP_SIZE_X + local_x;
    unsigned int global_x = group_x * WORKGROUP_SIZE_Y + local_x;
    unsigned int global_y = group_y * WORKGROUP_SIZE_X + local_y;
    if (global_x < K && global_y < M)
        buffer[local_id] = as[K * global_y + global_x];
    else
        buffer[local_id] = 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    local_id = local_x * WORKGROUP_SIZE_Y + local_y;
    global_x = group_y * WORKGROUP_SIZE_X + local_x;
    global_y = group_x * WORKGROUP_SIZE_Y + local_y;

    if (global_x < M && global_y < K)
        as_t[M * global_y + global_x] = buffer[local_id];
}

__kernel void calculate_prefix_parts(__global unsigned int *part_sums, unsigned int size) {
    const unsigned int id = get_global_id(0);
    part_sums[id * size] += part_sums[id * size + size / 2];
}

__kernel void calculate_prefix_sums(__global unsigned int *result, __global const unsigned int *part_sums,
                                    unsigned int size) {
    const unsigned int id = get_global_id(0) + 1;

    if (id & size) {
        result[id] += part_sums[(id - size) / size * size];
    }
}

__kernel void radix(__global const unsigned int *as, __global const unsigned int *prefix_sums,
                    __global unsigned int *tmp, unsigned int block_id, unsigned int size) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int group_id = get_group_id(0);
    const unsigned int local_id = get_local_id(0);
    const unsigned int work_group_size = get_local_size(0);

    unsigned int element_value = value(as[global_id], block_id);
    unsigned int offset = prefix_sums[element_value * size + group_id];

    for (int index = 0; index < local_id; ++index)
        offset += value(as[group_id * work_group_size + index], block_id) == element_value;

    tmp[offset] = as[global_id];
}

__kernel void reset(__global unsigned int *as) {
    as[get_global_id(0)] = 0;
}
