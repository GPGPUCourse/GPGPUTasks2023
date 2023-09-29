__kernel void matrix_multiplication_naive(__global float *as_gpu, __global float *bs_gpu, __global float *cs_gpu,
                                          unsigned int result_x_size, unsigned int result_common_size,
                                          unsigned int result_y_size) {
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    float result = 0;
    for (int c = 0; c < result_common_size; ++c) {
        result += as_gpu[i * result_common_size + c] * bs_gpu[c * result_y_size + j];
    }
    cs_gpu[i * result_y_size + j] = result;
}