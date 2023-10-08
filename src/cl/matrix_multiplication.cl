#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 6

#define GET(matrix, row, col, row_size, col_size) \
        (row < col_size && col < row_size) ? matrix[row * row_size + col] : 0.0f;


__kernel void
matrix_multiplication_naive(__global const float *a, __global const float *b, __global float *result, unsigned int M,
                            unsigned int K,
                            unsigned int N) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if ((x >= N) || (y >= M)) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += a[y * K + i] * b[i * N + x];
    }
    result[y * N + x] = sum;
}

#define TILE_SIZE 16

__kernel void
matrix_multiplication_local_memes(__global const float *a, __global const float *b, __global float *result,
                                  unsigned int M,
                                  unsigned int K,
                                  unsigned int N) {
    int j = get_global_id(1);
    int i = get_global_id(0);
    int local_j = get_local_id(1);
    int local_i = get_local_id(0);

    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;

    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        tileA[local_j][local_i] = GET(a, j, (tileK * TILE_SIZE + local_i), K, M)
        tileB[local_i][local_j] = GET(b, (tileK * TILE_SIZE + local_j), i, N, K)
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[local_j][k] * tileB[local_i][k];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (j < M && i < N) {
        result[j * N + i] = sum;
    }
}

#define THREAD_WORK 16

__kernel void
matrix_multiplication_task3_rows(__global const float *a, __global const float *b, __global float *result,
                                 const unsigned int M,
                                 const unsigned int K,
                                 const unsigned int N) {
    const int group_i = get_group_id(0);
    const int local_j = get_local_id(1);
    const int local_i = get_local_id(0);
    const int global_j = get_global_id(1);
    const int global_i = get_global_id(0);


    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE * THREAD_WORK + 1];

    float sum[THREAD_WORK];
    for (int i = 0; i < THREAD_WORK; ++i) {
        sum[i] = 0.0f;
    }

    for (int k = 0; k * TILE_SIZE < K; ++k) {
        tileA[local_j][local_i] = GET(a, global_j, k * TILE_SIZE + local_i, K, M)
        for (int i = 0; i < TILE_SIZE; i++) {
            tileB[local_j][local_i + i * THREAD_WORK] = GET(b, (k * TILE_SIZE + local_j),
                                                            (TILE_SIZE * THREAD_WORK * group_i + local_i +
                                                             i * THREAD_WORK), N, K)
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < TILE_SIZE; ++i) {
            float tmp = tileA[local_j][i];
            for (int j = 0; j < THREAD_WORK; ++j) {
                sum[j] += tmp * tileB[i][local_i * THREAD_WORK + j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < THREAD_WORK; ++i) {
        result[global_j * N + global_i * THREAD_WORK + i] = sum[i];
    }
}

__kernel void
matrix_multiplication_task3_columns(__global const float *a, __global const float *b, __global float *result,
                                    const unsigned int M,
                                    const unsigned int K,
                                    const unsigned int N) {
    const int group_j = get_group_id(1);
    const int local_j = get_local_id(1);
    const int local_i = get_local_id(0);
    const int global_j = get_global_id(1);
    const int global_i = get_global_id(0);


    __local float tileA[TILE_SIZE * THREAD_WORK][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    float sum[THREAD_WORK];
    for (int i = 0; i < THREAD_WORK; ++i) {
        sum[i] = 0.0f;
    }

    for (int k = 0; k * TILE_SIZE < K; ++k) {
        tileB[local_j][local_i] = GET(b, (k * TILE_SIZE + local_j), global_i, N, K)
        for (int i = 0; i < TILE_SIZE; i++) {
            tileA[local_j + i * THREAD_WORK][local_i] = GET(a, (TILE_SIZE * THREAD_WORK * group_j + local_j +
                                                                i * THREAD_WORK), (k * TILE_SIZE + local_i), K, M)
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < TILE_SIZE; ++i) {
            float tmp = tileB[i][local_i];
            for (int j = 0; j < THREAD_WORK; ++j) {
                sum[j] += tmp * tileA[local_j * THREAD_WORK + j][i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int i = 0; i < THREAD_WORK; ++i) {
        result[(global_j * THREAD_WORK + i) * N + global_i] = sum[i];
    }
}

#define NEW_THREAD_WORK 16

__kernel void
matrix_multiplication_task3_columns_no_memes(__global const float *a, __global const float *b, __global float *result,
                                             const unsigned int M,
                                             const unsigned int K,
                                             const unsigned int N) {
    const int group_j = get_group_id(1);
    const int local_j = get_local_id(1);
    const int local_i = get_local_id(0);
//    const int global_j = get_global_id(1);
    const int global_i = get_global_id(0);


    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    float sum[NEW_THREAD_WORK];
    for (int i = 0; i < NEW_THREAD_WORK; ++i) {
        sum[i] = 0.0f;
    }

    for (int k = 0; k * TILE_SIZE < K; ++k) {
        tileB[local_j][local_i] = GET(b, (k * TILE_SIZE + local_j), global_i, N, K)
        for (int work = 0; work < NEW_THREAD_WORK; ++work) {
            tileA[local_j][local_i] = GET(a, (TILE_SIZE * NEW_THREAD_WORK * group_j + local_j +
                                                                    work * TILE_SIZE), (k * TILE_SIZE + local_i), K, M)
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int i = 0; i < TILE_SIZE; ++i) {
                sum[work] += tileB[i][local_i] * tileA[local_j][i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);

        }
    }

    for (int i = 0; i < NEW_THREAD_WORK; ++i) {
        result[(TILE_SIZE * NEW_THREAD_WORK * group_j + local_j + TILE_SIZE * i) * N + global_i] = sum[i];
    }
}

__kernel void
matrix_multiplication_task3_rows_no_memes(__global const float *a, __global const float *b, __global float *result,
                                 const unsigned int M,
                                 const unsigned int K,
                                 const unsigned int N) {
    const int group_i = get_group_id(0);
    const int local_j = get_local_id(1);
    const int local_i = get_local_id(0);
    const int global_j = get_global_id(1);
    const int global_i = get_global_id(0);


    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    float sum[NEW_THREAD_WORK];
    for (int i = 0; i < NEW_THREAD_WORK; ++i) {
        sum[i] = 0.0f;
    }

    for (int k = 0; k * TILE_SIZE < K; ++k) {
        tileA[local_j][local_i] = GET(a, global_j, k * TILE_SIZE + local_i, K, M)
        for (int work = 0; work < NEW_THREAD_WORK; ++work) {
            tileB[local_j][local_i] = GET(b, (k * TILE_SIZE + local_j), (TILE_SIZE * NEW_THREAD_WORK * group_i + local_i +
                                                                            work * TILE_SIZE), N, K)

            barrier(CLK_LOCAL_MEM_FENCE);
            for (int i = 0; i < TILE_SIZE; ++i) {
                sum[work] += tileA[local_j][i] * tileB[i][local_i];

            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }

    for (int i = 0; i < NEW_THREAD_WORK; ++i) {
        result[global_j * N + TILE_SIZE * NEW_THREAD_WORK * group_i + local_i + i * TILE_SIZE] = sum[i];
    }
}
