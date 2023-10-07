#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif

// The code is almost identical to normal merge pass, but the pointer types are in different space

uint binsearch_lt(float x, __local const float *arr, uint len) {
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

uint binsearch_le(float x, __local const float *arr, uint len) {
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

void merge_pass(__local const float *src, __local float *dst, uint len, uint sorted_block_len, uint src_id) {
    if (src_id >= len) {
        return;
    }

    uint own_block = src_id / sorted_block_len;
    uint sib_block = own_block ^ 1;// *sib*ling

    uint own_block_start = own_block * sorted_block_len;
    uint sib_block_start = sib_block * sorted_block_len;

    //uint own_block_end = min(len, own_block_start + sorted_block_len);
    uint sib_block_end = min(len, sib_block_start + sorted_block_len);
    uint sib_block_len = sib_block_end - sib_block_start;
    __local const float *sib_block_src = src + sib_block_start;

    uint dst_id = src_id - own_block % 2 * sorted_block_len;
    float x = src[src_id];
    bool is_left = own_block % 2 == 0;
    if (is_left) {
        dst_id += binsearch_lt(x, sib_block_src, sib_block_len);
    } else {
        dst_id += binsearch_le(x, sib_block_src, sib_block_len);
    }

    dst[dst_id] = x;
}

__kernel void kmain(__global const float *src, __global float *dst, uint len) {
    __local float buf1[WORK_GROUP_SIZE];
    __local float buf2[WORK_GROUP_SIZE];

    uint global_id = get_global_id(0);
    uint local_id = get_local_id(0);
    uint local_start = global_id - local_id;
    uint local_len = min(len - local_start, (uint) WORK_GROUP_SIZE);

    buf1[local_id] = global_id < len ? src[global_id] : 0.0;
    buf2[local_id] = 0.0;
    barrier(CLK_LOCAL_MEM_FENCE);
    __local float *local_src = buf1;
    __local float *local_dst = buf2;

    for (uint sorted_size = 1; sorted_size < local_len; sorted_size *= 2) {
        merge_pass(local_src, local_dst, local_len, sorted_size, local_id);
        barrier(CLK_LOCAL_MEM_FENCE);
        __local float *tmp = local_src;
        local_src = local_dst;
        local_dst = tmp;
    }

    if (global_id < len) {
        dst[global_id] = local_src[local_id];
    }
}
