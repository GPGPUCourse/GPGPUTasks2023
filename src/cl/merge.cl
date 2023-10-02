uint binsearch_lt(
    float x,
    __global const float* arr,
    uint len
) {
    if (arr[0] >= x) {
        return 0;
    }
    uint l = 0;
    uint r = len;
    while (l + 1 < r) {
        uint m = l + (r - l) / 2;
        if (arr[m] < x) {
            l = m;
        } else {
            r = m;
        }
    }
    return r;
}

uint binsearch_le(
    float x,
    __global const float* arr,
    uint len
) {
    if (arr[0] > x) {
        return 0;
    }
    uint l = 0;
    uint r = len;
    while (l + 1 < r) {
        uint m = l + (r - l) / 2;
        if (arr[m] <= x) {
            l = m;
        } else {
            r = m;
        }
    }
    return r;
}

__kernel void merge(
    __global const float* src,
    __global float* dst,
    uint len,
    uint sorted_block_len
) {
    uint src_flat_id = get_global_id(0);
    if (src_flat_id >= len) {
        return;
    }

    uint own_block = src_flat_id / sorted_block_len;
    uint sib_block = own_block ^ 1; // *sib*ling

    uint own_block_start = own_block * sorted_block_len;
    uint sib_block_start = sib_block * sorted_block_len;

    uint own_block_end = min(len, own_block_start + sorted_block_len);
    uint sib_block_end = min(len, sib_block_start + sorted_block_len);

    uint dst_flat_id = src_flat_id - own_block % 2 * sorted_block_len;
    float x = src[src_flat_id];
    bool is_left = own_block % 2 == 0;
    if (is_left) {
        dst_flat_id += binsearch_lt(x, src + sib_block_start, sib_block_end - sib_block_start);
    } else {
        dst_flat_id += binsearch_le(x, src + sib_block_start, sib_block_end - sib_block_start);
    }

    dst[dst_flat_id] = x;
}
