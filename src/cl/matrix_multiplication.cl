#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif
#line 5

#define TILE_SIZE 16
#define THREADS 8

__kernel void matrix_multiplication_global_mem(__global const float* a,
                                               __global const float* b,
                                               __global float* c,
                                               unsigned M, unsigned K, unsigned N) {
    unsigned i = get_global_id(0);
    unsigned j = get_global_id(1);

    if (j >= M || i >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += a[j * K + k] * b[k * N + i];    

    c[j * N + i] = sum;
}

__kernel void matrix_multiplication_local_mem(__global const float* a,
                                              __global const float* b,
                                              __global float* c,
                                              unsigned M, unsigned K, unsigned N) {
    unsigned i = get_global_id(0);
    unsigned j = get_global_id(1);
    unsigned local_i = get_local_id(0);
    unsigned local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tileN = 0; tileN * TILE_SIZE < K; ++tileN) {

        tileA[local_j][local_i] = a[j * K + tileN * TILE_SIZE + local_i];
        tileB[local_j][local_i] = b[(tileN * TILE_SIZE + local_j) * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tileA[local_j][k] * tileB[k][local_i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (j < M && i < N)
        c[j * N + i] = sum;
}

__kernel void matrix_multiplication_local_mem_improved(__global const float* a,
                                                         __global const float* b,
                                                         __global float* c,
                                                         unsigned M, unsigned K, unsigned N) {
    unsigned i = get_global_id(0);
    unsigned j = get_global_id(1) * THREADS;
    unsigned local_i = get_local_id(0);
    unsigned local_j = get_local_id(1) * THREADS;

    __local float local_a[TILE_SIZE][TILE_SIZE];
    __local float local_b[TILE_SIZE][TILE_SIZE];
    __local float sum[TILE_SIZE][TILE_SIZE];

    for (int moved = 0; moved < THREADS; moved++)
        sum[local_j + moved][local_i] = 0.0f;

    for (int offset = 0; offset < K; offset += TILE_SIZE) {
        if (!local_j) {
            for (int k = 0; k < TILE_SIZE; k++)
                local_a[k][local_i] = a[(j + k) * K + offset + local_i];
            for (int k = 0; k < TILE_SIZE; k++)
                local_b[k][local_i] = b[(offset + k) * N + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            for (int moved = 0; moved < THREADS; moved++)
                sum[local_j + moved][local_i] += local_a[local_j + moved][k] * local_b[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int moved = 0; moved < THREADS; moved++)
        c[(j + moved) * N + i] = sum[local_j + moved][local_i];
}