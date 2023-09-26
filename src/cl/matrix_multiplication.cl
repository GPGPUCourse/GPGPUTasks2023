#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#define TILE_SIZE 16

__kernel void matrix_multiplication_naive(__global const float *as, __global const float *bs, __global float *cs, const unsigned int M, const unsigned int K, const unsigned int N)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        if (i < N && j < M) {
            sum += as[j * K + k] * bs[k * N + i];
        }
    }

    unsigned int c_index = j * N + i;
    if (i < N && j < M) {
        cs[c_index] = sum;
    }
}

__kernel void matrix_multiplication_local(__global const float *as, __global const float *bs, __global float *cs, const unsigned int M, const unsigned int K, const unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        int a_index = j * K + (tileK * TILE_SIZE + local_i);
        int b_index = i + N * (tileK * TILE_SIZE + local_j);

        tileA[local_j][local_i] = a_index < M * K ? as[a_index] : 0;
        tileB[local_j][local_i] = b_index < K * N ? bs[b_index] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < N && j < M) {
        cs[j * N + i] = sum;
    }
}
