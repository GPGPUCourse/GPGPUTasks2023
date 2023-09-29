#define WORK_GROUP_SIZE_X 16
#define WORK_GROUP_SIZE_Y 16


// coalesed local memory without bank conflicts
__kernel void matrix_transpose(__global float *as_gpu, __global float *as_t_gpu, unsigned int x_size, unsigned int y_size) {
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    const unsigned int buf_i = get_local_id(0);
    const unsigned int buf_j = get_local_id(1);
    
    __local float buf[WORK_GROUP_SIZE_X * WORK_GROUP_SIZE_Y];
    
    buf[buf_j * WORK_GROUP_SIZE_X + (buf_j + buf_i) % WORK_GROUP_SIZE_X] = as_gpu[j * x_size + i];
    barrier(CLK_LOCAL_MEM_FENCE);
    const unsigned int base_offset_i = i - buf_i;
    const unsigned int base_offset_j = j - buf_j;
    as_t_gpu[(base_offset_i + buf_j) * x_size  + base_offset_j + buf_i] = buf[buf_i * WORK_GROUP_SIZE_Y + (buf_j + buf_i) % WORK_GROUP_SIZE_X];
}

// // coalesed local memory
// __kernel void matrix_transpose(__global float *as_gpu, __global float *as_t_gpu, unsigned int x_size, unsigned int y_size) {
//     const unsigned int i = get_global_id(0);
//     const unsigned int j = get_global_id(1);
//     const unsigned int buf_i = get_local_id(0);
//     const unsigned int buf_j = get_local_id(1);
    
//     __local float buf[WORK_GROUP_SIZE_X * WORK_GROUP_SIZE_Y];
    
//     buf[buf_j * WORK_GROUP_SIZE_X + buf_i] = as_gpu[j * x_size + i];
//     barrier(CLK_LOCAL_MEM_FENCE);
//     const unsigned int base_offset_i = i - buf_i;
//     const unsigned int base_offset_j = j - buf_j;
//     as_t_gpu[(base_offset_i + buf_j) * x_size  + base_offset_j + buf_i] = buf[buf_i * WORK_GROUP_SIZE_Y + buf_j];
// }

// // naive local memory
// __kernel void matrix_transpose(__global float *as_gpu, __global float *as_t_gpu, unsigned int x_size, unsigned int y_size) {
//     const unsigned int i = get_global_id(0);
//     const unsigned int j = get_global_id(1);
//     // const unsigned int work_group_size_x = get_local_size(0);
//     // const unsigned int work_group_size_y = get_local_size(1);

//     __local float buf[WORK_GROUP_SIZE_X * WORK_GROUP_SIZE_Y];
//     const unsigned int buf_i = get_local_id(0);
//     const unsigned int buf_j = get_local_id(1);
//     buf[buf_i * WORK_GROUP_SIZE_Y + buf_j] = as_gpu[j * x_size + i];
//     barrier(CLK_LOCAL_MEM_FENCE);
//     as_t_gpu[i * y_size + j] = buf[buf_i * WORK_GROUP_SIZE_Y + buf_j];
// }


// // naive
// __kernel void matrix_transpose(__global float *as_gpu, __global float *as_t_gpu, unsigned int x_size, unsigned int y_size) {
//     const unsigned int i = get_global_id(0);
//     const unsigned int j = get_global_id(1);
//     as_t_gpu[i * x_size + j] = as_gpu[j * y_size + i];
// }