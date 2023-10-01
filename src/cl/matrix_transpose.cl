#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define TILE_SIZE 16

__kernel void matrix_transpose(__global float* as,
                               __global float* as_t,
                               unsigned int m,
                               unsigned int k)
{
    int global_i = get_global_id(0);
    int global_j = get_global_id(1);

    if (global_j >= m || global_i >= k) {
        return;
    }

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    if ((local_i >= TILE_SIZE) || (local_j >= TILE_SIZE)) {
        return;
    }

    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    tile[local_j][local_i] = as[global_j * k + global_i];

    barrier(CLK_LOCAL_MEM_FENCE);

    as_t[global_i * m + global_j] = tile[local_j][local_i];

}