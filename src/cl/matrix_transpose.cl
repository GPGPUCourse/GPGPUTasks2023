#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif

__kernel void matrix_transpose(__global const float *src, __global float *dst, uint src_size_x, uint src_size_y) {
    size_t src_gx = get_global_id(0);
    size_t src_gy = get_global_id(1);
    size_t src_lx = get_local_id(0);
    size_t src_ly = get_local_id(1);
    size_t src_flat_id = src_gy * src_size_x + src_gx;

    // Ячейки из одного столбца должны попадать в разные банки.
    // Тайл 16 на 17 даст непересекающееся распределение столбцов по банкам,
    // хотя оно и будет не таким очевидным, как в случае тайла 32 на 33.
    __local float tile[16][17];
    if (src_gx < src_size_x && src_gy < src_size_y) {
        tile[src_ly][src_lx] = src[src_flat_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    size_t dst_lx = src_ly;
    size_t dst_ly = src_lx;
    size_t dst_gx = src_gy + dst_ly - src_ly;
    size_t dst_gy = src_gx + dst_lx - src_lx;
    size_t dst_flat_id = dst_gy * src_size_y + dst_gx;

    if (dst_gx < src_size_y && dst_gy < src_size_x) {
        dst[dst_flat_id] = tile[dst_ly][dst_lx];
    }
}
