#ifndef WORK_GROUP_SIZE
    #include "clion_defines.cl"
#endif

__kernel void bitonic_global(__global float *as, uint len, uint half_bitonic_len, uint sorted_len) {
    uint gid = get_global_id(0);
    // 2: 0 1 2 3 4 5 -> 0 1 4 5 8 9
    // uint pos1 = gid / block_len * 2 * block_len + gid % block_len;
    uint i = 2 * gid - gid % sorted_len;
    uint j = i + sorted_len;
    if (j >= len) {
        return;
    }
    // If odd block, sort backwards
    if (gid / half_bitonic_len % 2 == 1) {
        uint t = i;
        i = j;
        j = t;
    }
    float x = as[i];
    float y = as[j];
    // printf("[%u] = %f, [%u] = %f", i, x, j, y);
    if (x > y) {
        as[i] = y;
        as[j] = x;
        // printf(", swapping");
    }
    // printf("\n");
}

__kernel void bitonic_local(__global float *as, uint len) {
    __local float buf[2 * WORK_GROUP_SIZE];
    uint gid = get_global_id(0);
    uint lid = get_local_id(0);

    buf[2 * lid + 0] = 2 * gid + 0 < len ? as[2 * gid + 0] : INFINITY;
    buf[2 * lid + 1] = 2 * gid + 1 < len ? as[2 * gid + 1] : INFINITY;

    for (uint half_bitonic_len = 1; half_bitonic_len < WORK_GROUP_SIZE; half_bitonic_len *= 2) {
        for (uint sorted_len = half_bitonic_len; sorted_len > 0; sorted_len /= 2) {
            uint i = 2 * lid - lid % sorted_len;
            uint j = i + sorted_len;
            if (lid / half_bitonic_len % 2 == 1) {
                uint t = i;
                i = j;
                j = t;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            float x = buf[i];
            float y = buf[j];
            if (x > y) {
                buf[i] = y;
                buf[j] = x;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (2 * gid + 0 < len) {
        as[2 * gid + 0] = buf[2 * lid + 0];
    }
    if (2 * gid + 1 < len) {
        as[2 * gid + 1] = buf[2 * lid + 1];
    }
}
