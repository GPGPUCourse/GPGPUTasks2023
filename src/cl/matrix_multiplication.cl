#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif
#line 5

__kernel void matrix_multiplication_base(__global const float* as,
                                    __global const float* bs,
                                    __global float* cs,
                                    unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if ((j < M) && (i < N)){
        float sum = 0;
        for (int k=0; k < K; ++k) {
            sum += as[j * K + k] * bs[k * N + i];
        }
        cs[j * N + i] = sum;
    }
}

#define TILE_SIZE 16
__kernel void matrix_multiplication_local_mem(__global const float* as,
                                      __global const float* bs,
                                      __global float* cs,
                                      unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    __local float tileA[TILE_SIZE * TILE_SIZE];
    __local float tileB[TILE_SIZE * TILE_SIZE];
    float sum = 0.0f;

    for (int tileK=0; tileK * TILE_SIZE < K; ++tileK) {
        int ax = TILE_SIZE * tileK + local_i;
        if ((j < M) && (ax < K)) {
            tileA[local_j * TILE_SIZE + local_i] = as[j * K + ax];
        } else {
            tileA[local_j * TILE_SIZE + local_i] = 0;
        }
        int by = TILE_SIZE * tileK + local_j;
        if ((by < K) && (i < N)) {
            tileB[local_j * TILE_SIZE + local_i] = bs[by * N + i];
        } else {
            tileB[local_j * TILE_SIZE + local_i] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k=0; k < TILE_SIZE; ++k) {
            sum += tileA[local_j * TILE_SIZE + k] * tileB[k * TILE_SIZE + local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if ((j < M) && (i < N)){
        cs[j * N + i] = sum;
    }
}

#define VALUES_PER_WORK_ITEM 8
__kernel void matrix_multiplication_busy(__global const float* as,
                                              __global const float* bs,
                                              __global float* cs,
                                              unsigned int M, unsigned int K, unsigned int N)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    __local float tileA[TILE_SIZE * TILE_SIZE * VALUES_PER_WORK_ITEM];
    __local float tileB[TILE_SIZE * TILE_SIZE];
    float sum[VALUES_PER_WORK_ITEM];
    for (int w=0; w<VALUES_PER_WORK_ITEM; ++w) {
        sum[w] = 0;
    }

    for (int tileK=0; tileK * TILE_SIZE < K; ++tileK) {
        for (int w=0; w<VALUES_PER_WORK_ITEM; ++w) {
            int ax = TILE_SIZE * tileK + local_i;
            int ay = j * VALUES_PER_WORK_ITEM + w;
            int local_ay = (local_j * VALUES_PER_WORK_ITEM + w);
            if ((ay < M) && (ax < K)) {
                tileA[local_ay * TILE_SIZE + local_i] = as[ay * K + ax];
            } else {
                tileA[local_ay * TILE_SIZE + local_i] = 0;
            }
        }
        int by = TILE_SIZE * tileK + local_j;
        if ((by < K) && (i < N)) {
            tileB[local_j * TILE_SIZE + local_i] = bs[by * N + i];
        } else {
            tileB[local_j * TILE_SIZE + local_i] = 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k=0; k < TILE_SIZE; ++k) {
            float tmp = tileB[k * TILE_SIZE + local_i];
            for (int w=0; w<VALUES_PER_WORK_ITEM; ++w) {
                sum[w] += tileA[(local_j * VALUES_PER_WORK_ITEM + w) * TILE_SIZE + k] * tmp;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w=0; w<VALUES_PER_WORK_ITEM; ++w) {
        int dst_y = j * VALUES_PER_WORK_ITEM + w;
        if ((dst_y < M) && (i < N)) {
            cs[dst_y * N + i] = sum[w];
        }
    }
}