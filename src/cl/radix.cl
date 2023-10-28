#ifndef WORK_GROUP_SIZE
    #include "clion_defines.cl"
#endif

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
        uint x = arr[gid];
        uint key = (x >> shift) & RADIX_MASK;
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

__kernel void radix_reorder(__global const uint *src, __global uint *dst, __global const uint *offsets, uint len,
                            uint shift) {
    __local uint data[WORK_GROUP_SIZE];
    __local uint buf1[RADIX_BASE * WORK_GROUP_SIZE];
    __local uint buf2[RADIX_BASE * WORK_GROUP_SIZE];
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint wid = get_group_id(0);

    data[lid] = gid < len ? src[gid] : RADIX_MASK;
    for (uint key = 0; key < RADIX_BASE; ++key) {
        buf1[key * WORK_GROUP_SIZE + lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint n_work_groups = (len + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;

    const uint my_x = data[lid];
    const uint my_key = (my_x >> shift) & RADIX_MASK;
    const uint my_off_lid = my_key * WORK_GROUP_SIZE + lid;
    if (gid < len) {
        buf1[my_off_lid] = 1;
    }
    const uint src_off_gid = my_key * n_work_groups + wid;
    const uint dst_off_wg = src_off_gid == 0 ? 0 : offsets[src_off_gid - 1];
    // printf("off[%u] = %u\n", src_off_gid, dst_off_wg);

    __local uint *loc_off_src = buf1;
    __local uint *loc_off_dst = buf2;
    for (uint step = 1; step < WORK_GROUP_SIZE; step *= 2) {
        barrier(CLK_LOCAL_MEM_FENCE);
        for (uint key = 0; key < RADIX_BASE; ++key) {
            const uint off_lid = key * WORK_GROUP_SIZE + lid;
            uint x = loc_off_src[off_lid];
            if (lid >= step) {
                x += loc_off_src[off_lid - step];
            }
            loc_off_dst[off_lid] = x;
        }
        __local uint *tmp = loc_off_src;
        loc_off_src = loc_off_dst;
        loc_off_dst = tmp;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint dst_off_lid = lid == 0 ? 0 : loc_off_src[my_off_lid - 1];
    const uint dst_off_gid = dst_off_wg + dst_off_lid;
    if (gid < len) {
        dst[dst_off_gid] = my_x;
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

#if false
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
#endif
