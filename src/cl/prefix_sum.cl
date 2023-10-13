#ifndef WORK_GROUP_SIZE
    #include "clion_defines.cl"
#endif

__kernel void cumsum_naive(__global const uint *src, __global uint *dst, uint len, uint ready_len) {
    uint gid = get_global_id(0);
    uint x = src[gid];
    if (gid >= ready_len) {
        x += src[gid - ready_len];
    }
    dst[gid] = x;
}
