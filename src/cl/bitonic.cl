#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6


__kernel void bitonic(__global float *as, const unsigned int block, const unsigned int depth) {
    int gid = get_global_id(0);
    int i = 2 * gid - gid % depth;
    int j = i + depth;

    int block_num = gid / block;
    if (block_num % 2 == 1) {
        int t = i;
        i = j;
        j = t;
    }

    if (as[i] > as[j]) {
        float t = as[j];
        as[j] = as[i];
        as[i] = t;
    }
}
