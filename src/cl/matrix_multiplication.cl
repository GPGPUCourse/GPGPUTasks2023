__kernel void matrix_multiplication(__global const float *a,
                                    __global const float *b,
                                    __global float *c,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N)
{
    int i = get_global_id(0) / N;
    int j = get_global_id(0) % N;
    float sum = 0.f;
    for(int k = 0; k < K; k++)
        sum += a[i * K + k] * b[k * N + j];
    c[i * N + j] = sum;
}