#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif

const size_t TILE_SIZE = 16;

__kernel void matrix_multiplication_0_naive(__global const float *matr_a, __global const float *matr_b,
                                            __global float *matr_c, uint M, uint K, uint N) {
    size_t gx = get_global_id(0);
    size_t gy = get_global_id(1);

    float sum = 0.0f;
    for (size_t k = 0; k < K; ++k) {
        if (gx < N && gy < M) {
            sum += matr_a[gy * K + k] * matr_b[k * N + gx];
        }
    }

    if (gx < N && gy < M) {
        matr_c[gy * N + gx] = sum;
    }
}

__kernel void matrix_multiplication_1_local(__global const float *matr_a, __global const float *matr_b,
                                            __global float *matr_c, uint M, uint K, uint N) {
    size_t wgx = get_group_id(0);
    size_t wgy = get_group_id(1);
    size_t gx = get_global_id(0);
    size_t gy = get_global_id(1);
    size_t lx = get_local_id(0);
    size_t ly = get_local_id(1);
    bool is_real = gx < N && gy < M;

    __local float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __local float tile_b[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;

    for (size_t tile_k = 0; tile_k < K; tile_k += TILE_SIZE) {
        if (is_real && tile_k + lx < N) {
            tile_a[ly][lx] = matr_a[gy * K + (tile_k + lx)];
        } else {
            tile_a[ly][lx] = 0.0f;
        }
        if (is_real && tile_k + ly < K) {
            tile_b[ly][lx] = matr_b[(tile_k + ly) * N + gx];
        } else {
            tile_b[ly][lx] = 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[ly][k] * tile_b[k][lx];
        }
    }

    if (is_real) {
        matr_c[gy * N + gx] = sum;
    }
}
