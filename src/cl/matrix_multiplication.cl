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

    __local float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __local float tile_b[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;

    for (size_t tile = 0; tile < K; tile += TILE_SIZE) {
        if (tile != 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        bool tile_a_real = row_g < M && tile + col_l < N;
        bool tile_b_real = col_g < N && tile + row_l < K;
        size_t tile_a_flat_id = row_g * K + (tile + col_l);
        size_t tile_b_flat_id = col_g + N * (tile + row_l);
        tile_a[row_l][col_l] = tile_a_real ? matr_a[tile_a_flat_id] : 0.0f;
        tile_b[row_l][col_l] = tile_b_real ? matr_b[tile_b_flat_id] : 0.0f;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t k = 0; k < TILE_SIZE; ++k) {
            sum += tile_a[row_l][k] * tile_b[k][col_l];
        }
    }

    if (col_g < N && row_g < M) {
        matr_c[row_g * N + col_g] = sum;
    }
}

const size_t RTS = TILE_SIZE / WORK_PER_THREAD;

__kernel void matrix_multiplication_3(__global const float *matr_a, __global const float *matr_b,
                                      __global float *matr_c, uint M, uint K, uint N) {
    size_t col_g = get_global_id(0);
    size_t col_l = get_local_id(0);
    size_t row_l = get_local_id(1);
    size_t row_g = get_group_id(1) * TILE_SIZE + row_l;

    __local float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __local float tile_b[TILE_SIZE][TILE_SIZE + 1];

    float sum[WORK_PER_THREAD] = {};

    for (size_t tile = 0; tile < K; tile += TILE_SIZE) {
        if (tile != 0) {
            barrier(CLK_LOCAL_MEM_FENCE);
        }

        size_t col_tiled = tile + col_l;
        size_t row_tiled = tile + row_l;

        for (size_t w = 0; w < TILE_SIZE; w += RTS) {
            bool tile_a_real = col_tiled < K && row_g + w < M;
            bool tile_b_real = col_g < N && row_tiled + w < K;
            size_t tile_a_flat_id = (row_g + w) * K + col_tiled;
            size_t tile_b_flat_id = (row_tiled + w) * N + col_g;
            tile_a[row_l + w][col_l] = tile_a_real ? matr_a[tile_a_flat_id] : 0.0f;
            tile_b[row_l + w][col_l] = tile_b_real ? matr_b[tile_b_flat_id] : 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t k = 0; k < TILE_SIZE; ++k) {
            float saved_b = tile_b[k][col_l];
            for (size_t w_id = 0; w_id < WORK_PER_THREAD; ++w_id) {
                sum[w_id] += tile_a[row_l + RTS * w_id][k] * saved_b;
            }
        }
    }

    for (size_t w_id = 0; w_id < WORK_PER_THREAD; ++w_id) {
        size_t row_w = row_g + RTS * w_id;
        bool is_real = row_w < M && col_g < N;
        if (is_real) {
            matr_c[row_w * N + col_g] = sum[w_id];
        }
    }
}
