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