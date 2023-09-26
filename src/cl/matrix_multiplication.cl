typedef unsigned int _uint;
__kernel void matrix_multiplication_simple(__global const float *a, __global const float *b, __global float *c,
                                           const _uint M, const _uint K, const _uint N) {
    _uint i = get_global_id(0);
    _uint j = get_global_id(1);

    if (i >= N || j >= M) {
        return;
    }

    float sum = 0;
    for (_uint k = 0; k < K; k++) {
        sum += a[j * K + k] * b[k * N + i];
    }

    c[j * N + i] = sum;
}

#define TILE_SIZE 16
__kernel void matrix_multiplication_tile(__global const float *a, __global const float *b, __global float *c,
                                         const _uint M, const _uint K, const _uint N) {
    _uint gid0 = get_global_id(0);
    _uint gid1 = get_global_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    _uint lid0 = get_local_id(0);
    _uint lid1 = get_local_id(1);

    float sum = 0.F;
    for (_uint number = 0; number < K; number += TILE_SIZE) {
        tileA[lid1][lid0] = (gid0 < N && gid1 < M) ? a[gid1 * K + number + lid0] : 0;
        tileB[lid1][lid0] = (gid0 < N && gid1 < M) ? b[number * N + lid1 * N + gid0] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);

        for (_uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[lid1][k] * tileB[k][lid0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gid0 < N && gid1 < M) {
        c[gid1 * N + gid0] = sum;
    }
}

#define THREAD_WORK 2
__kernel void matrix_multiplication_more_thread_work(__global const float *a, __global const float *b,
                                                     __global float *c, const _uint M, const _uint K, const _uint N) {
    _uint lid0 = get_local_id(0);
    _uint lid1 = get_local_id(1);

    _uint gid0 = get_global_id(0);
    _uint gid1 = get_group_id(1) * TILE_SIZE + lid1;

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    const _uint STEP = TILE_SIZE / THREAD_WORK;

    float sum[THREAD_WORK];
    for (_uint i = 0; i < THREAD_WORK; i++) {
        sum[i] = 0.F;
    }

    for (_uint number = 0; number < K; number += TILE_SIZE) {
        for (_uint i = 0; i < THREAD_WORK; i++) {
            _uint gid1n = gid1 + i * STEP;
            tileA[lid1 + i * STEP][lid0] = (gid0 < N && gid1n < M) ? a[gid1n * K + number + lid0] : 0;
            tileB[lid1 + i * STEP][lid0] = (gid0 < N && gid1n < M) ? b[(number + lid1 + i * STEP) * N + gid0] : 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (_uint k = 0; k < TILE_SIZE; k++) {
            float tmp = tileB[k][lid0];
            for (_uint i = 0; i < THREAD_WORK; i++) {
                sum[i] += tileA[lid1 + i * STEP][k] * tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (_uint i = 0; i < THREAD_WORK; i++) {
        gid1 = gid1 + i * STEP;
        if (gid0 < N && gid1 < M) {
            c[gid1 * N + gid0] = sum[i];
        }
    }
}
