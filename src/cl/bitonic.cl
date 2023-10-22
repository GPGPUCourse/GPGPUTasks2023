#ifdef __CLION_IDE__
#include <cl/clion_defines.cl>
#endif

#line 6

#define WG_SIZE 128

__kernel void bitonic_local(__global float* in, unsigned int block_size, unsigned int size, unsigned int length) {
    unsigned int local_id = get_local_id(0);
    unsigned int gid = get_global_id(0);

    __local float block[WG_SIZE];

    if (gid < length) {
        block[local_id] = in[gid];
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    int f = gid % (2 * size) < size;

    while (block_size >= 1) {

        if (gid % (2 * block_size) < block_size && gid + block_size < length) {
            float a = block[local_id];
            float b = block[local_id + block_size];

            if ((a > b) == f) {
                block[local_id] = b;
                block[local_id + block_size] = a;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        block_size >>= 1;
    }

    if (gid < length) {
        in[gid] = block[local_id];
    }
}


__kernel void bitonic(__global float* in, unsigned int block_size, unsigned int size, unsigned int length) {
    unsigned int gid = get_global_id(0);

    int f = gid % (2 * size) < size;

    if (gid % (2 * block_size) < block_size && gid + block_size < length) {
        float a = in[gid];
        float b = in[gid + block_size];
        if ((a > b) == f) {
            in[gid] = b;
            in[gid + block_size] = a;
        }
    }
}