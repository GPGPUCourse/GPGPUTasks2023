#ifndef WORK_GROUP_SIZE
    #include "clion_defines.cl"
#endif

__kernel void fill_zero(__global uint *dst, uint len) {
    const uint gid = get_global_id(0);
    if (gid < len) {
        dst[gid] = 0;
    }
}

__kernel void radix_count_local_t(__global const uint *arr, __global uint *dst, uint len, uint shift) {
    __local uint counts[RADIX_BASE];
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint wid = get_group_id(0);

    const uint n_work_groups = (len + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;

    for (uint radix_off = 0; radix_off < RADIX_BASE; radix_off += WORK_GROUP_SIZE) {
        const uint key = radix_off + lid;
        if (key < RADIX_BASE) {
            counts[key] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < len) {
        const uint x = arr[gid];
        const uint key = (x >> shift) & RADIX_MASK;
        atomic_add(&counts[key], 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint radix_off = 0; radix_off < RADIX_BASE; radix_off += WORK_GROUP_SIZE) {
        const uint key = radix_off + lid;
        // Already transposed, not coalesced
        const uint flat_dst = key * n_work_groups + wid;
        if (key < RADIX_BASE) {
            dst[flat_dst] = counts[key];
        }
    }
}

__kernel void radix_count_local(__global const uint *arr, __global uint *dst, uint len, uint shift) {
    __local uint counts[RADIX_BASE];
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint wid = get_group_id(0);

    const uint n_work_groups = (len + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;

    for (uint radix_off = 0; radix_off < RADIX_BASE; radix_off += WORK_GROUP_SIZE) {
        const uint key = radix_off + lid;
        if (key < RADIX_BASE) {
            counts[key] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (gid < len) {
        const uint x = arr[gid];
        const uint key = (x >> shift) & RADIX_MASK;
        atomic_add(&counts[key], 1);
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    for (uint radix_off = 0; radix_off < RADIX_BASE; radix_off += WORK_GROUP_SIZE) {
        const uint key = radix_off + lid;
        const uint flat_dst = key + RADIX_BASE * wid;
        if (key < RADIX_BASE) {
            dst[flat_dst] = counts[key];
        }
    }
}

__kernel void radix_count_t(__global const uint *arr, __global uint *dst, uint len, uint shift) {
    const uint gid = get_global_id(0);
    const uint wid = get_group_id(0);

    const uint n_work_groups = (len + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;

    if (gid < len) {
        const uint x = arr[gid];
        const uint key = (x >> shift) & RADIX_MASK;
        const uint flat_dst = key * n_work_groups + wid;
        atomic_add(&dst[flat_dst], 1);
    }
}

__kernel void radix_count(__global const uint *arr, __global uint *dst, uint len, uint shift) {
    const uint gid = get_global_id(0);
    const uint wid = get_group_id(0);

    const uint n_work_groups = (len + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;

    if (gid < len) {
        const uint x = arr[gid];
        const uint key = (x >> shift) & RADIX_MASK;
        const uint flat_dst = key + RADIX_BASE * wid;
        atomic_add(&dst[flat_dst], 1);
    }
}

uint binsearch_lt(uint x, __local const uint *arr, uint len, uint shift) {
    uint key = (arr[0] >> shift) & RADIX_MASK;
    x = (x >> shift) & RADIX_MASK;
    if (key >= x) {
        return 0;
    }
    uint l = 0;
    uint r = len;
    while (l + 1 < r) {
        uint m = l + (r - l) / 2;
        key = (arr[m] >> shift) & RADIX_MASK;
        if (key < x) {
            l = m;
        } else {
            r = m;
        }
    }
    return r;
}

uint binsearch_le(uint x, __local const uint *arr, uint len, uint shift) {
    uint key = (arr[0] >> shift) & RADIX_MASK;
    x = (x >> shift) & RADIX_MASK;
    if (key > x) {
        return 0;
    }
    uint l = 0;
    uint r = len;
    while (l + 1 < r) {
        uint m = l + (r - l) / 2;
        key = (arr[m] >> shift) & RADIX_MASK;
        if (key <= x) {
            l = m;
        } else {
            r = m;
        }
    }
    return r;
}

void merge_pass(__local const uint *src, __local uint *dst, uint sorted_block_len, uint src_id, uint shift) {
    const uint len = WORK_GROUP_SIZE;
    if (src_id >= len) {
        return;
    }

    const uint own_block = src_id / sorted_block_len;
    const uint sib_block = own_block ^ 1;// *sib*ling

    const uint own_block_start = own_block * sorted_block_len;
    const uint sib_block_start = sib_block * sorted_block_len;

    const uint sib_block_end = min(len, sib_block_start + sorted_block_len);
    const uint sib_block_len = sib_block_end - sib_block_start;
    __local const uint *sib_block_src = src + sib_block_start;

    uint dst_id = src_id - own_block % 2 * sorted_block_len;
    const uint x = src[src_id];
    const bool is_left = own_block % 2 == 0;
    if (is_left) {
        dst_id += binsearch_lt(x, sib_block_src, sib_block_len, shift);
    } else {
        dst_id += binsearch_le(x, sib_block_src, sib_block_len, shift);
    }

    dst[dst_id] = x;
}

__kernel void radix_reorder(__global const uint *src, __global uint *dst, __global const uint *offsets, uint len,
                            uint shift) {
    __local uint loc_val1[WORK_GROUP_SIZE];
    __local uint loc_val2[WORK_GROUP_SIZE];
    __local uint loc_off1[RADIX_BASE];
    __local uint loc_off2[RADIX_BASE];
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint wid = get_group_id(0);

    loc_val1[lid] = gid < len ? src[gid] : UINT_MAX;
    for (uint off = 0; off < RADIX_BASE; off += WORK_GROUP_SIZE) {
        uint flat = off + lid;
        if (flat < RADIX_BASE) {
            loc_off1[flat] = 0;
        }
    }

    __local uint *loc_val_src = loc_val1;
    __local uint *loc_val_dst = loc_val2;
    for (uint sorted = 1; sorted < WORK_GROUP_SIZE; sorted *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        merge_pass(loc_val_src, loc_val_dst, sorted, lid, shift);
        __local uint *tmp = loc_val_src;
        loc_val_src = loc_val_dst;
        loc_val_dst = tmp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint n_work_groups = (len + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;

    const uint my_x = loc_val_src[lid];
    const uint my_key = (my_x >> shift) & RADIX_MASK;
    const uint src_off_gid = my_key * n_work_groups + wid;
    const uint dst_off_wg = offsets[src_off_gid];
    atomic_add(&loc_off1[my_key], 1);

    __local uint *loc_off_src = loc_off1;
    __local uint *loc_off_dst = loc_off2;
    for (uint step = 1; step < WORK_GROUP_SIZE; step *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);

        for (uint off = 0; off < RADIX_BASE; off += WORK_GROUP_SIZE) {
            const uint flat = off + lid;
            if (flat < RADIX_BASE) {
                uint x = loc_off_src[flat];
                if (flat >= step) {
                    x += loc_off_src[flat - step];
                }
                loc_off_dst[flat] = x;
            }
        }

        __local uint *tmp = loc_off_src;
        loc_off_src = loc_off_dst;
        loc_off_dst = tmp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint loc_key_beg = my_key == 0 ? 0 : loc_off_src[my_key - 1];
    const uint dst_off_lid = lid - loc_key_beg;
    {
        const uint loc_key_end = loc_off_src[my_key];
        if (loc_key_beg > lid || lid >= loc_key_end) {
            printf("[ERROR]: %u <= %u < %u, key=%u\n", loc_key_beg, lid, loc_key_end, my_key);
        }
    }
    const uint dst_off_gid = dst_off_wg + dst_off_lid;
    if (gid < len) {
        // if (dst_off_gid >= len) {
        //     printf("[ERROR]: gid = %u, dst_off_gid = %u\n", gid, dst_off_gid);
        // }
        dst[dst_off_gid] = my_x;
    }
}

__kernel void matrix_transpose(__global const uint *src, __global uint *dst, uint src_size_x, uint src_size_y) {
    const uint src_col_g = get_global_id(0);
    const uint src_row_g = get_global_id(1);
    const uint src_col_l = get_local_id(0);
    const uint src_row_l = get_local_id(1);
    const uint src_flat_id = src_row_g * src_size_x + src_col_g;

    __local uint tile[TILE_SIZE][TILE_SIZE + 1];
    if (src_col_g < src_size_x && src_row_g < src_size_y) {
        tile[src_row_l][src_col_l] = src[src_flat_id];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint dst_col_l = src_row_l;
    const uint dst_row_l = src_col_l;
    const uint dst_col_g = src_row_g - src_row_l + dst_row_l;
    const uint dst_row_g = src_col_g - src_col_l + dst_col_l;
    const uint dst_flat_id = dst_row_g * src_size_y + dst_col_g;

    if (dst_col_g < src_size_y && dst_row_g < src_size_x) {
        dst[dst_flat_id] = tile[dst_row_l][dst_col_l];
    }
}

__kernel void cumsum_naive(__global const uint *src, __global uint *dst, uint len, uint step) {
    const uint gid = get_global_id(0);
    if (gid >= len) {
        return;
    }
    uint x = src[gid];
    if (gid >= step) {
        x += src[gid - step];
    }
    dst[gid] = x;
}

// Must ensure that array len is 2^n
__kernel void cumsum_sweep_up(__global uint *as, uint len, uint step) {
    uint gid = get_global_id(0);
    if (gid >= len) {
        return;
    }
    uint k = 2 * step * gid;
    as[k + 2 * step - 1] += as[k + step - 1];
}

__kernel void cumsum_sweep_down(__global uint *as, uint len, uint step) {
    uint gid = get_global_id(0);
    if (gid >= len) {
        return;
    }
    uint k = 2 * step * gid;
    uint left = as[k + step - 1];
    uint right = as[k + 2 * step - 1];
    as[k + step - 1] = right;
    as[k + 2 * step - 1] = left + right;
}

__kernel void shift_left(__global const uint *src, __global uint *dst, uint len) {
    uint gid = get_global_id(0);
    if (gid + 1 >= len) {
        return;
    }
    dst[gid] = src[gid + 1];
}
