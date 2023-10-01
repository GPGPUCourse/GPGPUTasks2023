#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#define TS 16
__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int M, unsigned int K) {
    unsigned int gx = get_global_id(0);
    unsigned int gy = get_global_id(1);
    unsigned int lx = get_local_id(0);
    unsigned int ly = get_local_id(1);

    __local float tile[TS][TS + 1];
    if (gx < K && gy < M)
        tile[ly][lx] = a[gy * K + gx];

    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned int n_gx = gx - lx + ly;
    unsigned int n_gy = gy - ly + lx;
    if (gx < K && gy < M) {
        at[n_gx * K + n_gy] = tile[lx][ly];
    }

}