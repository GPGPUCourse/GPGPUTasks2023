#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif


#define TS 16
#define WPT 4
__kernel void
matrix_multiplication(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K,
                      unsigned N) {
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int gx = get_global_id(0);
    int gy = get_group_id(1) * TS + ly;

    __local float tileA[TS][TS], tileB[TS][TS];
    const int RTS = TS / WPT;
    float sum[WPT];
    for (int w = 0; w < WPT; ++w) {
        sum[w] = 0.0f;
    }
    for (int tile_k = 0; tile_k * TS < K; ++tile_k) {
        for (int w = 0; w < WPT; ++w) {
            int gy_offset = gy + w * RTS;
            tileA[ly + w * RTS][lx] = gx < N && gy_offset < M ? a[gy_offset * K + tile_k * TS + lx] : 0;
            tileB[ly + w * RTS][lx] = gx < N && gy_offset < M ? b[(ly + tile_k * TS + w * RTS) * N + gx] : 0;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TS; ++k) {
            float tmpB = tileB[k][lx];
            for (int w = 0; w < WPT; ++w) {
                sum[w] += tileA[ly + w * RTS][k] * tmpB;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WPT; ++w) {
        c[(gy + w * RTS) * N + gx] = sum[w];
    }
}

__kernel void
matrix_multiplication_local(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K,
                            unsigned N) {
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    __local float tileA[TS][TS], tileB[TS][TS];

    float sum = 0.0f;
    for (int tile_k = 0; tile_k * TS < K; ++tile_k) {

        tileA[ly][lx] = gx < N && gy < M ? a[gy * K + tile_k * TS + lx] : 0;
        tileB[ly][lx] = gx < N && gy < M ? b[ly * N + gx + tile_k * TS * N] : 0;

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TS; ++k) {
            sum += tileA[ly][k] * tileB[k][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[gy * N + gx] = sum;
}


__kernel void
matrix_multiplication_naive(__global float *a, __global float *b, __global float *c, unsigned int M, unsigned int K,
                            unsigned N) {
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += a[j * K + k] * b[k * N + i];
    }
    c[j * N + i] = sum;
}