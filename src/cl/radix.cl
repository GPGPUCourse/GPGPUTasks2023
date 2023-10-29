#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#define WORKGROUP_SIZE 512
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
    , unsigned int as_length
    , unsigned int work_group_amount
) {
    unsigned int local_id = get_local_id(0);
    unsigned int work_group_id = get_group_id(0);
    unsigned int global_id = get_global_id(0);
    if (work_group_id == 0) {
        for (int i = 0; i < NUMBERS_AMOUNT; ++i) {
            bs[local_id - i > 0 ? counters[i * NUMBERS_AMOUNT] : 0] = as[global_id];
        }
    } else {
        for (int i = 0; i < NUMBERS_AMOUNT; ++i) {
            bs[prefixes[(i - 1) * NUMBERS_AMOUNT + work_group_id] + local_id -
            i > 0 ? counters[i * NUMBERS_AMOUNT + work_group_id] : 0] = as[global_id];
        }
    }
}

__kernel void local_sort
(
      __global const unsigned int *as
    , __global unsigned int *bs
    , const unsigned int bits
)
{

}