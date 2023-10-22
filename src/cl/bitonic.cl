#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif
#line 6
#define SWAP(as, lhs, rhs)                                                                                             \
    float tmp = as[rhs];                                                                                               \
    as[rhs] = as[lhs];                                                                                                 \
    as[lhs] = tmp;


__kernel void bitonic(__global float *as, unsigned int small_block_size, unsigned int big_block_size, unsigned int n) {
    unsigned int gid = get_global_id(0);
    unsigned int half_small_size = small_block_size / 2;
    unsigned int small_block_number = gid / half_small_size;
    unsigned int index = small_block_number * small_block_size + gid % half_small_size;

    unsigned int big_block_number = index / big_block_size;
    int sign = (big_block_number & 1) ? -1: 1;

    if (index < n && sign * as[index] > sign * as[index + half_small_size]) {
        SWAP(as, index, index + half_small_size);
    }
}
