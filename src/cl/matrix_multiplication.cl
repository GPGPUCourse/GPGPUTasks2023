#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define TILE_SIZE 16
#define THREAD_WORK 16

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
    int global_j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    int group_j = get_group_id(1);


    __local float tile_a[TILE_SIZE * THREAD_WORK][TILE_SIZE + 1];
    __local float tile_b[TILE_SIZE][TILE_SIZE + 1];

    float sum[THREAD_WORK] = {0.0f};

    for (int i = 0; i < THREAD_WORK; ++i) {
        sum[i] = 0.0f;
    }

    for (int shift = 0; shift * TILE_SIZE < K; ++shift) {
        tile_b[local_j][local_i] = bs[(TILE_SIZE * shift + local_j) * N + global_i];

        for (int i = 0; i < TILE_SIZE; ++i) {
            if ((TILE_SIZE * THREAD_WORK * group_j + local_j +
                 i * THREAD_WORK) >= M) {
                tile_a[local_j + i * THREAD_WORK][local_i] = 0.0f;
            }else {
                tile_a[local_j + i * THREAD_WORK][local_i] =
                        as[(TILE_SIZE * THREAD_WORK * group_j + local_j + i * THREAD_WORK) * K +
                           (shift * TILE_SIZE + local_i)];
            }

        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = 0; s < TILE_SIZE; ++s) {
            float tmp = tile_b[s][local_i];
            for (int w = 0; w < THREAD_WORK; ++w) {
                sum[w] += tmp * tile_a[local_j * THREAD_WORK + w][s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (global_j * THREAD_WORK>= N) {
        return;
    }
    for (int w = 0; w < THREAD_WORK; ++w) {
        cs[(global_j * THREAD_WORK + w) * N + global_i] = sum[w];
    }
}