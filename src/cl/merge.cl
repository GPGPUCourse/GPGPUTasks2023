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
    if (global_id > n) {
        return;
    }
    unsigned local_size = get_local_size(0);
    if (local_size > size) {
        unsigned local_id = get_local_id(0);
        unsigned inner_id = local_id / size;
        unsigned group_id = get_group_id(0);
        unsigned left = group_id * local_size + inner_id * size;
        unsigned right = left + size;
        unsigned mid = (left + right) / 2;
        float element = src[global_id];

        if (global_id == 1) {
            printf("global_id: %d\nlocal_id: %d\ninner_id: %d\ngroup_id: %d\nleft: %d\nright: %d\nmid: %d\nelement: "
                   "%f\n",
                   global_id, local_id, inner_id, group_id, left, right, mid, element);
        }

        if (global_id < mid) {
            unsigned offset = bin_search_ne(element, mid, right, src) - mid;
            res[left + local_id % size + offset] = src[global_id];
            if (global_id == 1) {
                printf("offset: %d\n", offset);
                printf("index to1: %d\n", left + local_id % size + offset);
            }
        } else {
            unsigned offset = bin_search_eq(element, left, mid, src) - left;
            res[left + local_id % size - size / 2 + offset] = src[global_id];
            if (global_id == 1) {
                printf("offset: %d\n", offset);
                printf("index to2: %d\n", left + local_id % size - size + offset);
            }
        }
        if (global_id == 1)
            printf("++++++++++++++++++++++++\n");

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
            res[mid + global_id % size + offset] = src[global_id];
        }
    }
}