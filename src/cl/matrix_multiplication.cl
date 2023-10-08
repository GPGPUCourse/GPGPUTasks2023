#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>
#include <libgpu/opencl/cl/common.cl>

#endif

#line 6

__kernel void matmul_naive(__global float *A, __global float *B, __global float *C,
                           unsigned int M, unsigned int K, unsigned int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < M && j < K) {
        float sum_buffer = 0;
        for (int k = 0; k < K; k++) {
            sum_buffer = fma(A[j * K + k], B[k * N + i], sum_buffer);
        }

        C[j * N + i] = sum_buffer;
    }
}

#define TILE_SIZE 16
__kernel void matmul_local(__global const float *a, __global const float *b, __global float *c,
                           unsigned int M, unsigned int K, unsigned int N) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);


    __local float ldsA[TILE_SIZE][TILE_SIZE + 1];
    __local float ldsB[TILE_SIZE][TILE_SIZE + 1];
    float sum_buffer = 0;
    for (unsigned step = 0; step * TILE_SIZE < K; step++) {
        ldsA[local_y][local_x] = a[y * K + local_x + step * TILE_SIZE];
        ldsB[local_y][local_x] = b[(local_y + step * TILE_SIZE) * N + x];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned index = 0; index < TILE_SIZE; index++) {
            sum_buffer = fma(ldsA[local_y][index], ldsB[index][local_x], sum_buffer);
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (x < N && y < M) {
        c[y * N + x] = sum_buffer;
    }
}

#define WPT 8
#define RTS 2
__kernel void matmul_wpt(const __global float *a, const __global float *b, __global float *c,
                                      const unsigned int M, const unsigned int K, const unsigned int N) {
    unsigned x = get_local_id(0);
    unsigned y = get_local_id(1);

    int global_x = TILE_SIZE * get_group_id(0) + x;
    int global_y = TILE_SIZE * get_group_id(1) + y;

    __local float ldsA[TILE_SIZE][TILE_SIZE + 1];
    __local float ldsB[TILE_SIZE][TILE_SIZE + 1];

    if (x < M && y < K) {
        float sum_buffer[WPT];
        for (int w = 0; w < WPT; ++w) {
            sum_buffer[w] = 0.0f;
        }
        const int numTiles = K / TILE_SIZE;
        for (int t = 0; t < numTiles; ++t) {
            for (int w = 0; w < WPT; ++w) {
                const int tiledRow = TILE_SIZE * t + x;
                const int tiledCol = TILE_SIZE * t + y;
                ldsA[y + w * RTS][x] = a[(global_y + w * RTS) * K + tiledRow];
                ldsB[y + w * RTS][x] = b[(tiledCol + w * RTS) * M + global_x];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            for (int k = 0; k < TILE_SIZE; ++k) {
                for (int w = 0; w < WPT; ++w) {
                    sum_buffer[w] = fma(ldsA[y + w * RTS][k], ldsB[k][x], sum_buffer[w]);
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        for (int w = 0; w < WPT; ++w) {
            c[(global_y + w * RTS) * M + global_x] = sum_buffer[w];
        }
    }
}




