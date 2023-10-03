unsigned bin_search_ne(const float x, unsigned left, unsigned right, __global const float *src) {
    while (left + 1 < right) {
        unsigned mid = (left + right) / 2;
        if (src[mid] >= x) {
            right = mid;
        } else {
            left = mid;
        }
    }
    return left;
}

unsigned bin_search_eq(const float x, unsigned left, unsigned right, __global const float *src) {
    while (left + 1 < right) {
        unsigned mid = (left + right) / 2;
        if (src[mid] > x) {
            right = mid;
        } else {
            left = mid;
        }
    }
    return left;
}


__kernel void merge(__global const float *src, __global float *res, const unsigned size, const unsigned n) {
    unsigned global_id = get_global_id(0);
    unsigned local_size = get_local_size(0);
    if (local_size > size) {
        unsigned local_id = get_local_id(0);
        unsigned inner_id = local_id / size;
        unsigned group_id = get_group_id(0);
        unsigned left = group_id * local_size + inner_id * size;
        unsigned right = left + size;
        unsigned mid = (left + right) / 2;
        float element = src[global_id];
        if (global_id < mid) {
            unsigned offset = bin_search_ne(element, mid, right, src) - mid;
            res[left + local_id % size + offset] = src[global_id];
        } else {
            unsigned offset = bin_search_eq(element, left, mid, src) - left;
            res[left + local_id % size + offset] = src[global_id];
        }
    } else {
        unsigned group_id = global_id / size;
        unsigned left = group_id * size;
        unsigned right = left + size;
        unsigned mid = (left + right) / 2;
        float element = src[global_id];
        if (global_id < mid) {
            unsigned offset = bin_search_ne(element, mid, right, src) - mid;
            res[left + global_id % size + offset] = src[global_id];
        } else {
            unsigned offset = bin_search_eq(element, left, mid, src) - left;
            res[left + global_id % size + offset] = src[global_id];
        }
    }
}