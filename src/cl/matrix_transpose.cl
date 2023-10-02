#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif

__kernel void matrix_transpose(__global const float *src, __global float *dst, uint src_size_x, uint src_size_y) {
    size_t src_col_g = get_global_id(0);
    size_t src_row_g = get_global_id(1);
    size_t src_col_l = get_local_id(0);
    size_t src_row_l = get_local_id(1);
    size_t src_flat_id = src_row_g * src_size_x + src_col_g;

    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    if (src_col_g < src_size_x && src_row_g < src_size_y) {
        tile[src_row_l][src_col_l] = src[src_flat_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    size_t dst_col_l = src_row_l;
    size_t dst_row_l = src_col_l;
    size_t dst_col_g = src_row_g - src_row_l + dst_row_l;
    size_t dst_row_g = src_col_g - src_col_l + dst_col_l;
    size_t dst_flat_id = dst_row_g * src_size_y + dst_col_g;

    if (dst_col_g < src_size_y && dst_row_g < src_size_x) {
        dst[dst_flat_id] = tile[dst_row_l][dst_col_l];
    }
}
