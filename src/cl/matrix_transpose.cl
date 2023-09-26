#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

#define TILE_SIZE 16

__kernel void matrix_transpose(__global const float *a, __global float *res, unsigned m, unsigned k) {
    const int i = get_global_id(1);
    const int j = get_global_id(0);

    __local float tile[TILE_SIZE][TILE_SIZE];

    const int li = get_local_id(1);
    const int lj = get_local_id(0);

    // Считываем в столбец.
    tile[lj][li] = a[i * k + j];

    // Дожидаемся остальных.
    barrier(CLK_LOCAL_MEM_FENCE);

    const int gi = get_group_id(1);
    const int gj = get_group_id(0);

    // Записываем элемент строки в нужную позицию.
    res[gj * TILE_SIZE * m + gi * TILE_SIZE + li * m + lj] = tile[li][lj];
}
