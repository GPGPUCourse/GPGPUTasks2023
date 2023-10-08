#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define TILE_SIZE 16
#define THREAD_WORK 4

__kernel void matrix_multiplication(__global float* as,
                                    __global float* bs,
                                    __global float* cs,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += as[j * K + k] * bs[k * N + i];
    }
    cs[j * N + i] = sum;
}

__kernel void matrix_multiplication2(__global float* as,
                                    __global float* bs,
                                    __global float* cs,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N)
{
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if (global_i >= M || global_j > N) {
        return;
    }
    if (local_i > TILE_SIZE || local_j > TILE_SIZE) {
        return;
    }

    __local float tile_a[TILE_SIZE][TILE_SIZE + 1];
    __local float tile_b[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;
    for (int shift = 0; shift * TILE_SIZE < K; ++shift) {
        tile_a[local_j][local_i] = as[global_j * K + TILE_SIZE * shift + local_i];
        tile_b[local_j][local_i] = bs[(TILE_SIZE * shift + local_j) * N + global_i];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = 0; s < TILE_SIZE; ++s) {
            sum += tile_a[local_j][s] * tile_b[s][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    cs[global_j * N + global_i] = sum;
}


__kernel void matrix_multiplication3(__global float* as,
                                     __global float* bs,
                                     __global float* cs,
                                     unsigned int M,
                                     unsigned int K,
                                     unsigned int N)
{
    int global_i = get_global_id(0);
    int global_j = get_global_id(1) * THREAD_WORK;
    int local_i = get_local_id(0);
    int local_j = get_local_id(1) * THREAD_WORK;

    __local float sum[TILE_SIZE][TILE_SIZE];
    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    for (int w = 0; w < THREAD_WORK; ++w) {
        sum[local_j + w][local_i] = 0.0f;
    }

    for (int shift = 0; shift * TILE_SIZE < K; ++shift) {
        if (!local_j) {
            for (int i = 0; i < TILE_SIZE; ++i) {
                tile_a[i][local_i] = as[(global_j + i) * K + shift * TILE_SIZE + local_i];
                tile_b[i][local_i] = bs[(shift * TILE_SIZE + i) * N + global_i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int j = 0; j < TILE_SIZE; ++j) {
            float tmp = tile_b[j][local_i];
            for (int w = 0; w < THREAD_WORK; ++w)
                sum[local_j + w][local_i] += tile_a[local_j + w][j] * tmp;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < THREAD_WORK; ++w)
        cs[(global_j + w) * N + global_i] = sum[local_j + w][local_i];
}