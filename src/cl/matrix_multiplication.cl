#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif

__kernel void matrix_multiplication_0_naive(__global const float *as, __global const float *bs, __global float *cs,
                                            uint M, uint K, uint N) {
    size_t gx = get_global_id(0);
    size_t gy = get_global_id(1);

    float sum = 0.0f;
    for (size_t k = 0; k < K; ++k) {
        if (gx < N && gy < M) {
            sum += as[gy * K + k] * bs[k * N + gx];
        }
    }

    if (gx < N && gy < M) {
        cs[gy * N + gx] = sum;
    }
}