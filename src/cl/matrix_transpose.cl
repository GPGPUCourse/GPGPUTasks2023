__kernel void matrix_transpose(__global float *as_gpu, __global float *as_t_gpu, unsigned int x_size, unsigned int y_size) {
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    float tmp = as_gpu[j * y_size + i];
    as_t_gpu[i * x_size + j] = tmp;
}