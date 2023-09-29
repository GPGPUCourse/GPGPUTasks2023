__kernel void matrix_multiplication_1(const __global float *a, const __global float *b, __global float *c,
                                      const unsigned int M, const unsigned int K, const unsigned int N) {
    // TODO
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    if (i < M && j < K) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += a[j * K + k] * b[k * N + i];
        }
        c[j * N + i] = sum;
    }
}

#define TILE_SIZE 16
__kernel void matrix_multiplication_2(const __global float *a, const __global float *b, __global float *c,
                                      const unsigned int M, const unsigned int K, const unsigned int N) {
    // TODO
    const uint i = get_local_id(0);
    const uint j = get_local_id(1);

    const int global_i = TILE_SIZE * get_group_id(0) + i;
    const int global_j = TILE_SIZE * get_group_id(1) + j;

    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    if (i < M && j < K) {
        float sum = 0.0f;
        const int numTiles = K / TILE_SIZE;
        for (int t = 0; t < numTiles; ++t) {
            const int tiledRow = TILE_SIZE * t + i;
            const int tiledCol = TILE_SIZE * t + j;
            tileA[j][i] = a[global_j * K + tiledRow];
            tileB[j][i] = b[tiledCol * M + global_i];

            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k = 0; k < TILE_SIZE; ++k) {
                sum += tileA[j][k] * tileB[k][i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        c[global_j * M + global_i] = sum;
    }
}

#define TILE_SIZE 16
#define WORK_PER_THREAD 8
#define RTS TILE_SIZE / WORK_PER_THREAD
__kernel void matrix_multiplication_3(const __global float *a, const __global float *b, __global float *c,
                                      const unsigned int M, const unsigned int K, const unsigned int N) {
    // TODO
    const uint i = get_local_id(0);
    const uint j = get_local_id(1);

    const int global_i = TILE_SIZE * get_group_id(0) + i;
    const int global_j = TILE_SIZE * get_group_id(1) + j;

    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    if (i < M && j < K) {
        float sum[WORK_PER_THREAD];
        for (int w = 0; w < WORK_PER_THREAD; ++w) {
            sum[w] = 0.0f;
        }
        const int numTiles = K / TILE_SIZE;
        for (int t = 0; t < numTiles; ++t) {
            for (int w = 0; w < WORK_PER_THREAD; ++w) {
                const int tiledRow = TILE_SIZE * t + i;
                const int tiledCol = TILE_SIZE * t + j;
                tileA[j + w * RTS][i] = a[(global_j + w * RTS) * K + tiledRow];
                tileB[j + w * RTS][i] = b[(tiledCol + w * RTS) * M + global_i];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k = 0; k < TILE_SIZE; ++k) {
                for (int w = 0; w < WORK_PER_THREAD; ++w) {
                    sum[w] += tileA[j + w * RTS][k] * tileB[k][i];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (int w = 0; w < WORK_PER_THREAD; ++w) {
            c[(global_j + w * RTS) * M + global_i] = sum[w];
        }
    }
}