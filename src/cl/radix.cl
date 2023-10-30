#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#define WORKGROUP_SIZE 128
#define BITS_AMOUNT 4
// NUMBERS_AMOUNT = 2 ^ BITS_AMOUNT
#define NUMBERS_AMOUNT 16
#define SIZE_OF_ELEMENT 32

__kernel void radix
(
      __global const unsigned int *as
    , __global const unsigned int *counters
    , __global const unsigned int *prefixes
    , __global unsigned int *bs
    , const unsigned int offset
    , const unsigned int n
    , unsigned int work_group_amount
) {
    unsigned int local_id = get_local_id(0);
    unsigned int work_group_id = get_group_id(0);
    unsigned int global_id = get_global_id(0);
    unsigned char bits = (as[global_id] >> offset) % NUMBERS_AMOUNT;
    unsigned int index = local_id;
    if (global_id >= n)
    {
        return;
    }
    for (int i = 0; i < bits; ++i)
    {
        index -= counters[i * work_group_amount + work_group_id];
    }

    if (work_group_id == 0)
    {
        if (bits != 0)
        {
            index += prefixes[work_group_amount * (bits - 1) + work_group_amount - 1];
        }
    }
    else
    {
        if (bits == 0)
        {
            index += prefixes[work_group_id - 1];
        }
        else
        {
            index += prefixes[work_group_amount * bits + work_group_id - 1];
        }
    }

    bs[index] = as[global_id];
}

__kernel void small_merge_sort(__global unsigned int *a,
                               __global unsigned int *b,
                               unsigned int shift) {
    int local_id = get_local_id(0);
    int local_size = get_local_size(0);
    int group_id = get_group_id(0);
    int begin = group_id * local_size;
    __global unsigned int *src = a + begin;
    __global unsigned int *dst = b + begin;
    int counter = 0;
    int mask = ((1 << BITS_AMOUNT) - 1) << shift;
    for (int block = 1; block <= local_size / 2; block <<= 1, ++counter) {
        unsigned int value = src[local_id];
        int k = local_id / (2 * block);
        int i = local_id % (2 * block);
        int j;
        if (i < block) {
            int l0 = k * (2 * block) + block - 1;
            int left = l0;
            int right = (k + 1) * (2 * block);
            if (left > local_size)
                left = local_size;
            if (right > local_size)
                right = local_size;
            while (right - left > 1) {
                int middle = (left + right) / 2;
                if ((src[middle] & mask) < (value & mask)) {
                    left = middle;
                } else {
                    right = middle;
                }
            }
            j = i + (left - l0);
        } else {
            int l0 = k * (2 * block) - 1;
            int left = l0;
            int right = k * (2 * block) + block;
            while (right - left > 1) {
                int middle = (left + right) / 2;
                if ((src[middle] & mask) <= (value & mask)) {
                    left = middle;
                } else {
                    right = middle;
                }
            }
            j = (i - block) + (left - l0);
        }
        dst[k * (2 * block) + j] = value;
        __global unsigned int *tmp = src;
        src = dst;
        dst = tmp;

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (counter % 2 == 0) {
        b[local_id] = a[local_id];
    }
}

// Naive realisation
__kernel void counters
(
      __global const unsigned int *as
    , __global unsigned int *counters
    , const unsigned int offset
    , const unsigned int n
    , const unsigned int workgroup_amount
)
{
    unsigned int global_id = get_global_id(0);
    unsigned int workgroup_id = get_group_id(0);

    if (global_id >= n) {
        return;
    }

    unsigned int bits = (as[global_id] >> offset) % NUMBERS_AMOUNT;
    atomic_add(&counters[workgroup_id * NUMBERS_AMOUNT + bits], 1);
}

// Naive parallel realisation
__kernel void prefixes
        (
                  __global const unsigned int *input
                , __global unsigned int *result
                , const unsigned int n
                , const unsigned int offset
        )
{
    unsigned int global_id = get_global_id(0);
    unsigned int workgroup_id = get_group_id(0);

    if (global_id >= n) {
        return;
    }
    unsigned int pow2 = 1 << (offset - 1);
    if (global_id >= pow2)
    {
        result[global_id] = input[global_id - pow2] + input[global_id];
    }
    else
    {
        result[global_id] = input[global_id];
    }
}

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

// Это реализация вопроса с лекции, как воркгруппой 16*16 обработать тайл 32*32, однако для моей карты тайл 16*16 будет
// обеспечивать coalesced memory access, так как размер кэш линии у меня = 64, а максимальный размер воркгруппы 512
// Также, чтобы разрешить bank-conflicts, размер локального буфера сделал 32*33 (как указывалось на лекции, чтобы
// элементы одного столбца лежали в разных банках)
__kernel void matrix_transpose(const __global unsigned int *a, __global unsigned int *at, unsigned int m, unsigned int k) {
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

__kernel void zero(__global unsigned int *a, unsigned int n)
{
    unsigned int global_id = get_global_id(0);
    if (global_id >= n)
    {
        return;
    }
    a[global_id] = 0;
}

__kernel void copy(__global unsigned int *source, __global unsigned int *target, unsigned int size)
{
    unsigned int global_id = get_global_id(0);
    if (global_id >= size)
    {
        return;
    }

    target[global_id] = source[global_id];
}