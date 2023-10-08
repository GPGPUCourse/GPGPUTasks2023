#ifdef __CLION_IDE__

#include "clion_defines.cl"

#endif

#line 6

#define INFINITY 1.79769e+308

#define GET(a, n, i) ((i) < n ? a[(i)] : INFINITY)

#define BIN_SEARCH(a, left_index, right_index, target, left_part)                       \
    unsigned int cur_left = left_index;                                                 \
    unsigned int cur_right = right_index;                                               \
    while (cur_right > cur_left) {                                                      \
        if ((left_part && (GET(a, n, (cur_left + cur_right) / 2) == target))            \
        || GET(a, n, (cur_left + cur_right) / 2) > target) {                            \
            cur_right = (cur_left + cur_right) / 2;                                     \
        } else {                                                                        \
            cur_left = (cur_left + cur_right) / 2 + 1;                                  \
        }                                                                               \
    }                                                                                   \
    unsigned int result = cur_left;


__kernel void merge_sort
    (
          __global const float *a
        , __global float *b
        , const unsigned int merge_size
        , const unsigned int n
    ) {
    const int global_i = get_global_id(0);
    unsigned int left = global_i / merge_size * merge_size;
    unsigned int right = left + merge_size < n ? left + merge_size : n;
    unsigned int center_of_array = (left + right) / 2;

    if (global_i >= n) {
        return;
    }

    if (global_i < center_of_array)
    {
        BIN_SEARCH(a, center_of_array, right, a[global_i], true)
        b[global_i - left + result - merge_size / 2] = a[global_i];
    } else {
        BIN_SEARCH(a, left, center_of_array, a[global_i], false)
        b[global_i - center_of_array + result] = a[global_i];
    }
}
