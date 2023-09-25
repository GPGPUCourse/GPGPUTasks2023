#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

__kernel void matrix_transpose(__global const float* as, __global float* as_t, uint M, uint K)
{
    size_t g_x = get_global_id(0);
    size_t g_y = get_global_id(1);
    size_t idx = g_y * K + g_x;
    size_t idx_t = g_x * M + g_y;
    if (g_x < M && g_y < K) {
        as_t[idx_t] = as[idx];
    }
}
