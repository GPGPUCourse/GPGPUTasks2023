#ifndef WORK_GROUP_SIZE
    #include "clion_defines.cl"
#endif

__kernel void bitonic_global(__global float *as, uint len, uint bitonic_len, uint sorted_len) {
    uint gid = get_global_id(0);
    // 2: 0 1 2 3 4 5 -> 0 1 4 5 8 9
    // uint pos1 = gid / block_len * 2 * block_len + gid % block_len;
    uint i = 2 * gid - gid % sorted_len;
    uint j = i + sorted_len;
    if (j >= len) {
        return;
    }
    // If odd block, sort backwards
    if (gid / bitonic_len % 2 == 1) {
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
