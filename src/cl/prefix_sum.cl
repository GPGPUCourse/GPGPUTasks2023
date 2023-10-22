#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif

#line 6

__kernel void Reduce(__global unsigned int *as, unsigned int block_size, unsigned int n) {
    unsigned int gid = get_global_id(0);
    unsigned int index = gid * 2 * block_size - 1;
    if (index + 2 * block_size < n) {
        as[index + 2 * block_size] += as[index + block_size];
    }
}

__kernel void Summarize(__global unsigned int *as, __global unsigned int *ans, unsigned int block_size) {
    unsigned int gid = get_global_id(0);
    unsigned int block_index = ((gid / block_size) * 2 + 1) * block_size;
    ans[block_index + gid % block_size - 1] += as[block_index - 1];
}