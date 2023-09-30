#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

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