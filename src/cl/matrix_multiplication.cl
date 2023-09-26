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
    for (_uint tile_number = 0; tile_number * TILE_SIZE < K; tile_number++) {
        if (gid0 < N && gid1 < M) {
            tileA[lid1][lid0] = a[gid1 * K + tile_number * TILE_SIZE + lid0];
            tileB[lid1][lid0] = b[tile_number * TILE_SIZE * N + lid1 * N + gid0];
        } else {
            tileA[lid1][lid0] = 0.;
            tileB[lid1][lid0] = 0.;
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (_uint k = 0; k < TILE_SIZE; k++) {
            sum += tileA[lid1][k] * tileB[k][lid0];
        }
    }

    if (gid0 < N && gid1 < M) {
        c[gid1 * N + gid0] = sum;
    }
}