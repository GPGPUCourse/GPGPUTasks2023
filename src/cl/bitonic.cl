#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif
#line 5

__kernel void bitonic(__global float *as,
                      unsigned int block_size,
                      unsigned int gap) {
    unsigned int gid = get_global_id(0);
    unsigned int block_index = gid / (block_size / 2);
    unsigned int subblock_index = (gid - block_index * block_size / 2) / gap;
    unsigned int from = block_index * block_size / 2 + subblock_index * gap + gid;
    // printf("%d %d %d %d %d %d\n", block_size, gap, gid, block_index, subblock_index, from);
    float low = as[from];
    float high = as[from + gap];
    if (((low > high) && (block_index % 2 == 0)) || ((low < high) && (block_index % 2 == 1))) {
        as[from] = high;
        as[from + gap] = low;
    }
}
