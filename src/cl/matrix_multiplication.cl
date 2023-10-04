// vim: syntax=c

__kernel void matrix_multiplication_naive(
    __global const float * a,
    __global const float * b,
    __global float * c,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    float sum = 0;
    for (unsigned int i = 0; i < K; ++i) {
        sum += a[y * K + i] * b[i * N + x];
    }

    c[y * N + x] = sum;
}
