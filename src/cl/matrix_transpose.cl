#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif
#line 5

#define TILE_SIZE 16
__kernel void matrix_transpose(__global const float* as,
                               __global float* as_t,
                               unsigned int M, unsigned int K)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    int local_ij = (local_i + local_j) % TILE_SIZE;
    __local float tile[TILE_SIZE][TILE_SIZE];

    if ((i < K) && (j < M)) {
        tile[local_j][local_ij] = as[j * K + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int dst_x = j - local_j + local_i;
    int dst_y = i - local_i + local_j;
    if ((dst_x < M) && (dst_y < K)) {
        as_t[dst_y * M + dst_x] = tile[local_i][local_ij];
    }
}