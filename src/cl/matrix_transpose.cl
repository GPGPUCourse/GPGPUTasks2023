#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 6

#define TILE_SIZE 32
#define HALF_SIZE TILE_SIZE / 2

// Использую макросы, так как лямбды OpenCL не поддерживает

#define STORE_WITH_OFFSETS(os_x, os_y, os_x_tile, os_y_tile) \
        if ((os_x + global_x < k) && (os_y + global_y < m)) {                                                \
            (tile)[(local_y + os_y_tile)][(local_x + os_x_tile)] = (a)[(os_y + global_y) * (k) + global_x + os_x]; \
        }

#define WRITE_WITH_OFFSETS(os_x, os_y, os_x_tile, os_y_tile) \
        if ((os_x + global_x < k) && (os_y + global_y < m)) {                                                 \
            (at)[(os_x + global_x) * (m) + global_y + os_y] = (tile)[(local_y + os_y_tile)][(local_x + os_x_tile)]; \
        }

// Здесь реализован coalesced memory access, так как размер воркгруппы больше, чем 16*16 не ставится, то приходится
// извращаться таким образом (плюс это был вопрос с лекции: воркгруппой 16*16 обработать тайл 32*32). Так как кэш линия
// подгружает 32 float значения, то логично их использовать все, поэтому размер локального буфера 32*32.
// Также, чтобы разрешить bank-conflicts, размер локального буфера сделал 32*33 (как указывалось на лекции, чтобы
// элементы одного столбца лежали в разных банках)
__kernel void matrix_transpose(__global float *a, __global float *at, unsigned int m, unsigned int k) {
    int global_x = get_global_id(0);
    int global_y = get_global_id(1);

    int group_x = get_group_id(0);
    int group_y = get_group_id(1);

    int offset_x = HALF_SIZE * group_x;
    int offset_y = HALF_SIZE * group_y;

    int second_offset_x = offset_x + HALF_SIZE;
    int second_offset_y = offset_y + HALF_SIZE;

    __local float tile[TILE_SIZE][TILE_SIZE + 1];
    int local_x = get_local_id(0);
    int local_y = get_local_id(1);

    STORE_WITH_OFFSETS(offset_x, offset_y, 0, 0)
    STORE_WITH_OFFSETS(second_offset_x, offset_y, HALF_SIZE, 0)
    STORE_WITH_OFFSETS(offset_x, second_offset_y, 0, HALF_SIZE)
    STORE_WITH_OFFSETS(second_offset_x, second_offset_y, HALF_SIZE, HALF_SIZE)

    barrier(CLK_LOCAL_MEM_FENCE);

    WRITE_WITH_OFFSETS(offset_x, offset_y, 0, 0)
    WRITE_WITH_OFFSETS(second_offset_x, offset_y, HALF_SIZE, 0)
    WRITE_WITH_OFFSETS(offset_x, second_offset_y, 0, HALF_SIZE)
    WRITE_WITH_OFFSETS(second_offset_x, second_offset_y, HALF_SIZE, HALF_SIZE)
}
