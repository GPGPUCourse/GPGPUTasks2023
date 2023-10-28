#ifndef COUNTER_SIZE
    #define COUNTER_SIZE 4
#endif

#ifndef MASK
    #define MASK (COUNTER_SIZE - 1)
#endif

unsigned bin_search_ne(const unsigned x, unsigned left, unsigned right, __global const unsigned *src,
                       const unsigned degree) {
    while (left < right) {
        unsigned mid = (left + right) / 2;
        if (((src[mid] >> degree) & MASK) < x) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

unsigned bin_search_eq(const unsigned x, unsigned left, unsigned right, __global const unsigned *src,
                       const unsigned degree) {
    while (left < right) {
        unsigned mid = (left + right) / 2;
        if (((src[mid] >> degree) & MASK) <= x) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__kernel void merge(__global const unsigned *src, __global unsigned *res, const unsigned degree, const unsigned size,
                    const unsigned n) {
    unsigned global_id = get_global_id(0);
    if (global_id >= n) {
        return;
    }
    unsigned local_size = get_local_size(0), left, mid, right, start;
    unsigned local_id = get_local_id(0);
    unsigned group_id = get_group_id(0);
    left = group_id * local_size + (local_id / size) * size;
    right = left + size;
    mid = (left + right) / 2;
    start = left + global_id % size;
    unsigned old_element = src[global_id];
    unsigned element = (old_element >> degree) & MASK;
    if (global_id < mid) {
        unsigned offset = bin_search_ne(element, mid, right, src, degree) - mid;
        res[start + offset] = old_element;
    } else {
        unsigned offset = bin_search_eq(element, left, mid, src, degree) - left;
        res[start - size / 2 + offset] = old_element;
    }
}