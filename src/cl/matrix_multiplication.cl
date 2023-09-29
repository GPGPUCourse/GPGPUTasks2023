#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 6

__kernel void
matrix_multiplication_naive(__global const float *a, __global const float *b, __global float *result, unsigned int M,
                            unsigned int K,
                            unsigned int N) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    if ((x >= N) || (y >= M)) {
        return;
    }

    float sum = 0.0f;
    for (int i = 0; i < K; ++i) {
        sum += a[y * K + i] * b[i * N + x];
    }
    result[y * N + x] = sum;
}


__kernel void
matrix_multiplication_local_memes(__global const float *a, __global const float *b, __global float *result, unsigned int M,
                            unsigned int K,
                            unsigned int N) {
    int x = get_global_id(0);
    int y = get_global_id(0);

}

