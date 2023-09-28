#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void naive(__global const float* a,
                    __global const float* b,
                    __global float* c,
                    uint M, uint K, uint N) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < K; ++k)
        sum += a[j * K + k] * b[k * N + i];
    c[j * N + i] = sum;
}

__kernel void local_memory(__global const float* a,
                           __global const float* b,
                           __global float* c,
                           uint M, uint K, uint N) {
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    float sum = 0.0f;
    for (int tileK = 0; tileK < K; tileK += TILE_SIZE) {
        tileA[local_j][local_i] = a[j * K + (tileK + local_i)];
        tileB[local_j][local_i] = b[(tileK + local_j) * N + i];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tileA[local_j][k] * tileB[k][local_i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[j * N + i] = sum;
}

__kernel void local_memory_with_more_work_per_thread(__global const float* a,
                                         __global const float* b,
                                         __global float* c,
                                         uint M, uint K, uint N) {
    const size_t RTS = TILE_SIZE / WORK_PER_THREAD;

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    int i = get_global_id(0);
    int j = get_group_id(1) * TILE_SIZE + local_j;

    __local float tileA[TILE_SIZE][TILE_SIZE + 1];
    __local float tileB[TILE_SIZE][TILE_SIZE + 1];

    float sum[WORK_PER_THREAD];
    for (uint w = 0; w < WORK_PER_THREAD; w++) {
        sum[w] = 0.F;
    }

    for (int tileK = 0; tileK < K; tileK += TILE_SIZE) {
        size_t tile_i = tileK + local_i;
        size_t tile_j = tileK + local_j;

        for (size_t w = 0; w < TILE_SIZE; w += RTS) {
            tileA[local_j + w][local_i] = tile_i < K && j + w < M
                                          ? a[(j + w) * K + tile_i]
                                          : 0.0f;
            tileB[local_j + w][local_i] = i < N && tile_j + w < K
                                          ? b[(tile_j + w) * N + i]
                                          : 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; ++k) {
            float cache_tileB = tileB[k][local_i];
            for (int w = 0; w < WORK_PER_THREAD; ++w) {
                sum[w] += tileA[local_j + w * RTS][k] * cache_tileB;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (size_t w = 0; w < WORK_PER_THREAD; ++w) {
        size_t temp_j = j + RTS * w;
        if (temp_j < M && i < N)
            c[temp_j * N + i] = sum[w];
    }
}

