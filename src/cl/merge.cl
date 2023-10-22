#ifdef __CLION_IDE__

#include <cl/clion_defines.cl>
#include <math.h>

#endif

#line 6

__kernel void merge(__global float *in, __global float *out, unsigned int length, unsigned int block_size) {
    unsigned int gid = get_global_id(0);

    if (gid < length) {
        unsigned int a_start = gid / (block_size << 1) * (block_size << 1);
        unsigned int a_end = min(a_start + block_size, length);
        unsigned int b_start = a_end;
        unsigned int b_end = min(b_start + block_size, length);

        if (b_start >= length) {
            in[gid] = out[gid];
        } else {

            unsigned int a_pos = gid - a_start;
            int l = max((int) a_pos - (int) (b_end - b_start), 0) - 1;
            int r = min(block_size, a_pos);
            while (l + 1 < r) {
                int m = (l + r) / 2;
                if (in[a_start + m] <= in[b_start + a_pos - m - 1]) {
                    l = m;
                } else {
                    r = m;
                }
            }

            unsigned int a = a_start + r;
            unsigned int b = b_start + a_pos - r;

            if (a < a_end && (b >= b_end || in[a] <= in[b])) {
                out[gid] = in[a];
            } else {
                out[gid] = in[b];
            }
        }
    }
}

