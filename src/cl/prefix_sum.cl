#ifndef WORK_GROUP_SIZE
    #include "clion_defines.cl"
#endif

__kernel void cumsum_naive(__global const uint *src, __global uint *dst, uint len, uint step) {
    uint gid = get_global_id(0);
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
