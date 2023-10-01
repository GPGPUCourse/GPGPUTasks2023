#define TILE_SIZE 32
#define WORK_PER_THREAD 4

__kernel void matrix_multiplication_naive(__global const float* a, __global const float* b, __global float* c, unsigned int M, unsigned int K, unsigned int N)
{
    int j = get_global_id(0) / N, i = get_global_id(0) % N;
    float sum = 0.0f;

    if (i < N && j < M) {
        for (int k = 0; k < K; ++k) {
            sum += a[j * K + k] * b[k * N + i];
        }
        c[j * N + i] = sum;
    }
}

__kernel void matrix_multiplication_local_mem(__global const float* a, __global const float* b, __global float* c, unsigned int M, unsigned int K, unsigned int N)
{
    int j = get_global_id(1), i = get_global_id(0);
    int local_j = get_local_id(1), local_i = get_local_id(0);
    float sum = 0.0f;

    __local float blockA[TILE_SIZE][TILE_SIZE];
    __local float blockB[TILE_SIZE][TILE_SIZE];

    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        if (j < M && TILE_SIZE * tileK + local_i < K) {
            blockA[local_j][local_i] = a[j * K + TILE_SIZE * tileK + local_i];
        }
        if (TILE_SIZE * tileK + local_j < K && i < N) {
            blockB[local_j][local_i] = b[(TILE_SIZE * tileK + local_j) * N + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            if (j < M && TILE_SIZE * tileK + k < K && TILE_SIZE * tileK + k < K && i < N)
            sum += blockA[local_j][k] * blockB[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if (j < M && i < N) {
        c[j * N + i] = sum;
    }
}

__kernel void matrix_multiplication_more_work(__global const float* a, __global const float* b, __global float* c, unsigned int M, unsigned int K, unsigned int N)
{
    int j = get_global_id(1), i = get_global_id(0);
    int local_j = get_local_id(1), local_i = get_local_id(0);

    float sum[WORK_PER_THREAD];
    for (int h = 0; h < WORK_PER_THREAD; ++h) {
        sum[h] = 0.0f;
    }

    __local float blockA[TILE_SIZE * WORK_PER_THREAD][TILE_SIZE];
    __local float blockB[TILE_SIZE][TILE_SIZE];

    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        for (int h = 0; h < WORK_PER_THREAD; ++h) {
            if (j * WORK_PER_THREAD + h < M && TILE_SIZE * tileK + local_i < K) {
                blockA[local_j * WORK_PER_THREAD + h][local_i] = a[(j * WORK_PER_THREAD + h) * K + TILE_SIZE * tileK + local_i];
            }
        }
        if (TILE_SIZE * tileK + local_j < K && i < N) {
            blockB[local_j][local_i] = b[(TILE_SIZE * tileK + local_j) * N + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            float tmp = blockB[k][local_i];
            for (int h = 0; h < WORK_PER_THREAD; ++h) {
            if (j * WORK_PER_THREAD + h < M && TILE_SIZE * tileK + k < K && TILE_SIZE * tileK + k < K && i < N)
                sum[h] += blockA[local_j * WORK_PER_THREAD + h][k] * tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int h = 0; h < WORK_PER_THREAD; ++h) {
        if (j * WORK_PER_THREAD + h < M && i < N) {
            c[(j * WORK_PER_THREAD + h) * N + i] = sum[h];
        }
    }
}