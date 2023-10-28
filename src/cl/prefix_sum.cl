#ifdef __CLION_IDE__

#include <cl/clion_defines.cl>

#endif

#line 6

__kernel void prefix_sum(__global const unsigned int *in,
                         __global unsigned int *out,
                         unsigned int length,
                         unsigned int level) {
    const unsigned int i = get_global_id(0);
    if (i < length) {
        if (((i + 1) >> level) & 1) {
            out[i] += in[((i + 1) >> level) - 1];
        }
    }
}

__kernel void prefix_sum_other(__global const unsigned int *in,
                               __global unsigned int *out,
                               unsigned int length) {
    const unsigned int i = get_global_id(0);
    if (i < length) {
        out[i] = in[2 * i] + in[2 * i + 1];
    }
}