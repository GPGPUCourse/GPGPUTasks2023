#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define TILE_SIZE 32

__kernel void matrix_transpose(__global const float* matrix,
                               __global float* transposed_matrix,
                               uint m, uint k) {
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    // Я не понял смысл вообще юзать TILE_SIZE, если делать + 1
    // Только если делать сдвига, который займёт всё равно больше памяти
    __local float tile[TILE_SIZE / 2][(TILE_SIZE / 2) + 1];
    size_t local_i = get_local_id(0);
    size_t local_j = get_local_id(1);

    tile[local_i][local_j] = matrix[j * k + i];
    barrier(CLK_LOCAL_MEM_FENCE);

    // И не понял смысл этого действа, если наверху и куалесд доступ есть, и не должно быть банк конфликтов из-за + 1
//    float temp = tile[local_j][local_i];
//    tile[local_j][local_i] = tile[local_i][local_j];
//    tile[local_i][local_j] = temp;
//    barrier(CLK_LOCAL_MEM_FENCE);

    size_t group_id_i = get_group_id(0);
    size_t group_id_j = get_group_id(1);
    size_t group_size_i = get_local_size(0);
    size_t group_size_j = get_local_size(1);

    size_t temp_i = group_id_i * group_size_i + local_j;
    size_t temp_j = group_id_j * group_size_j + local_i;
    transposed_matrix[temp_i * m + temp_j] = tile[local_j][local_i];
}
