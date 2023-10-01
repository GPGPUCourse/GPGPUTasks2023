#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

#define TILE_SIZE 16
#define WORK_PER_THREAD 16

__kernel void matrix_multiplication_naive(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < K; k++) {
        sum += a[j * K + k] * b[k * N + i];
    }
    c[j * N + i] = sum;
}

__kernel void matrix_multiplication_local(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    int group_i = get_group_id(0);
    int group_j = get_group_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < K; tileK++) {
        tileA[local_j][local_i] = a[(group_j * TILE_SIZE + local_j) * K + tileK * TILE_SIZE + local_i];
        tileB[local_j][local_i] = b[(tileK * TILE_SIZE + local_j) * K + group_i * TILE_SIZE + local_i];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[j * N + i] = sum;
}

__kernel void matrix_multiplication_local_work(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K, unsigned int N)
{
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    int group_i = get_group_id(0);
    int group_j = get_group_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum[TILE_SIZE];
    for (int k = 0; k < TILE_SIZE; k++)
        sum[k] = 0.0f;

    for (int tileK = 0; tileK * TILE_SIZE < K; tileK++) {
        for (int k = 0; k < TILE_SIZE; k++) {
            tileA[k][local_i] = a[(group_j * TILE_SIZE + k) * K + tileK * TILE_SIZE + local_i];
            tileB[k][local_i] = b[(tileK * TILE_SIZE + k) * N + group_i * TILE_SIZE + local_i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            float tmp = tileB[k][local_i];
            for (int w = 0; w < TILE_SIZE; w++) {
                sum[w] += tmp * tileA[w][k];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int k = 0; k < TILE_SIZE; k++)
        c[(group_j * TILE_SIZE + k) * N + group_i * TILE_SIZE + local_i] = sum[k];
}
