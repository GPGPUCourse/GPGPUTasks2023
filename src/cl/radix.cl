#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#define WORKGROUP_SIZE 512
#define BITS_AMOUNT 4
// NUMBERS_AMOUNT = 2 ^ BITS_AMOUNT
#define NUMBERS_AMOUNT (1 << BITS_AMOUNT)
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
    if (work_group_id == 0)
    {
        if (bits != 0)
        {
            index += prefixes[work_group_amount * bits + NUMBERS_AMOUNT - 1] - counters[work_group_amount * (bits - 1)];
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
            index += prefixes[work_group_amount * bits + work_group_id - 1] - counters[work_group_amount * (bits - 1) + work_group_id];
        }
    }

    bs[index] = as[global_id];
}

// Complexity is O(2*WORKGROUP_SIZE)
__kernel void local_bubble_sort (
          __global unsigned int *a
        , const unsigned int n
        , const unsigned bit_offset
) {
    unsigned int local_id = get_local_id(0);
    unsigned int work_group_id = get_group_id(0);
    unsigned int id = work_group_id * WORKGROUP_SIZE + 2 * local_id;
    for (int i = 0; i < 2 * WORKGROUP_SIZE; ++i)
    {
        if ((id < n) && (id < (work_group_id + 1) * WORKGROUP_SIZE))
        {
            if ((a[id] >> bit_offset) % NUMBERS_AMOUNT > (a[id + 1] >> bit_offset) % NUMBERS_AMOUNT)
            {
                float buf = a[id + 1];
                a[id + 1] = a[id];
                a[id] = buf;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        if ((id < n) && (id < (work_group_id + 1) * WORKGROUP_SIZE) && (id > work_group_id * WORKGROUP_SIZE))
        {
            if ((a[id - 1] >> bit_offset) % NUMBERS_AMOUNT > (a[id] >> bit_offset) % NUMBERS_AMOUNT)
            {
                float buf = a[id];
                a[id] = a[id - 1];
                a[id - 1] = buf;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void counters
(
      __global const unsigned int *as
    , __global unsigned int *counters
    , const unsigned int offset
    , const unsigned int n
)
{
    unsigned int global_id = get_global_id(0);

}