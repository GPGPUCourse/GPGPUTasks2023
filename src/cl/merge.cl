unsigned bin_search_ne(const float x, unsigned left, unsigned right, __global const float *src) {
    while (left < right) {
        unsigned mid = (left + right) / 2;
        if (src[mid] < x) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

unsigned bin_search_eq(const float x, unsigned left, unsigned right, __global const float *src) {
    while (left < right) {
        unsigned mid = (left + right) / 2;
        if (src[mid] <= x) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    return left;
}

__kernel void merge(__global const float *src, __global float *res, const unsigned size, const unsigned n) {
    unsigned global_id = get_global_id(0);
    if (global_id >= n) {
        return;
    }
    unsigned local_size = get_local_size(0), left, mid, right, start;
    if (local_size >= size) {
        unsigned local_id = get_local_id(0);
        unsigned group_id = get_group_id(0);
        left = group_id * local_size + (local_id / size) * size;
    } else {
        left = (global_id / size) * size;
    }
    right = left + size;
    mid = (left + right) / 2;
    start = left + global_id % size;
    float element = src[global_id];
    if (global_id < mid) {
        unsigned offset = bin_search_ne(element, mid, right, src) - mid;
        res[start + offset] = element;
    } else {
        unsigned offset = bin_search_eq(element, left, mid, src) - left;
        res[start - size / 2 + offset] = element;
    }
}