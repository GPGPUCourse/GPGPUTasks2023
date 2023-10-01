#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define TILE_SIZE 16
#define THREAD_WORK 2

__kernel void matmul1_naive(__global float *a, __global float *b, __global float *c,
                            unsigned int M, unsigned int K, unsigned int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    
    if (i >= N || j >= M) 
        return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) 
        sum += a[j * K + k] * b[k * N + i];

    c[j * N + i] = sum;
}

__kernel void matmul2_local_memory(__global float *a, __global float *b, __global float *c,
                                   unsigned int M, unsigned int K, unsigned int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        tileA[local_j][local_i] = a[j * K + (tileK * TILE_SIZE + local_i)];
        tileB[local_j][local_i] = b[(tileK * TILE_SIZE + local_j) * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tileA[local_j][k] * tileB[k][local_i];

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[j * N + i] = sum;
}

__kernel void matmul3_local_memory_more_per_thread(__global float *a, __global float *b, __global float *c,
                                                   unsigned int M, unsigned int K, unsigned int N) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE * THREAD_WORK];

    float sum[THREAD_WORK];
    for (int i = 0; i < THREAD_WORK; ++i)
        sum[i] = 0;

    const int num_tiles = N / THREAD_WORK;

    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {

        for (int w = 0; w < THREAD_WORK; ++w)
            tileB[local_j][local_i * THREAD_WORK + w] = b[(tileK * TILE_SIZE + local_j) * N + w * num_tiles + i];

        tileA[local_j][local_i] = a[j * K + (tileK * TILE_SIZE + local_i)];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            float tmp = tileA[local_j][k];
            for (int w = 0; w < THREAD_WORK; ++w)
                sum[w] += tmp * tileB[k][local_i * THREAD_WORK + w];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

    }
    for (int w = 0; w < THREAD_WORK; ++w)
        c[j * N + w * num_tiles + i] = sum[w];
}