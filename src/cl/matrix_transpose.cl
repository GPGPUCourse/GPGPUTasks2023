#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#define TS 16
__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int M, unsigned int K) {
    const unsigned int gx = get_group_id(0) * TS;
    const unsigned int gy = get_group_id(1) * TS;
    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    __local float buf[TS][TS + 1];

    if (gx + lx < M && gy + ly < K) {
        buf[ly][lx] = a[(gy + ly) * M + (gx + lx)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx + ly < M && gy + lx < K) {
        at[(gx + ly) * K + (gy + lx)] = buf[lx][ly];
    }

}