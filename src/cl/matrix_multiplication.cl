#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif

__kernel void matrix_multiplication_1(__global const float *matr_a, __global const float *matr_b,
                                      __global float *matr_c, uint M, uint K, uint N) {
    size_t col_g = get_global_id(0);
    size_t row_g = get_global_id(1);

    float sum = 0.0f;
    for (size_t k = 0; k < K; ++k) {
        if (col_g < N && row_g < M) {
            sum += matr_a[row_g * K + k] * matr_b[k * N + col_g];
        }
    }

    if (col_g < N && row_g < M) {
        matr_c[row_g * N + col_g] = sum;
    }
}

__kernel void matrix_multiplication_2(__global const float *matr_a, __global const float *matr_b,
                                      __global float *matr_c, uint M, uint K, uint N) {
    size_t col_g = get_global_id(0);
    size_t row_g = get_global_id(1);
    size_t col_l = get_local_id(0);
    size_t row_l = get_local_id(1);
    bool is_real = col_g < N && row_g < M;

    __local float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __local float tile_b[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;

    for (size_t tile_k = 0; tile_k < K; tile_k += TILE_SIZE) {
        if (tile_k != 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (is_real && tile_k + col_l < N) {
            tile_a[row_l][col_l] = matr_a[row_g * K + (tile_k + col_l)];
        } else {
            tile_a[row_l][col_l] = 0.0f;
        }
        if (is_real && tile_k + row_l < K) {
            tile_b[row_l][col_l] = matr_b[(tile_k + row_l) * N + col_g];
        } else {
            tile_b[row_l][col_l] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[row_l][k] * tile_b[k][col_l];
        }
    }

    if (is_real) {
        matr_c[row_g * N + col_g] = sum;
    }
}

__kernel void matrix_multiplication_3(__global const float *matr_a, __global const float *matr_b,
                                      __global float *matr_c, uint M, uint K, uint N) {
    size_t col_g = get_global_id(0);
    size_t row_g = get_global_id(1);

    float sum = 0.0f;
    for (size_t k = 0; k < K; ++k) {
        if (col_g < N && row_g < M) {
            sum += matr_a[row_g * K + k] * matr_b[k * N + col_g];
        }
    }

    if (col_g < N && row_g < M) {
        matr_c[row_g * N + col_g] = sum;
    }
}
