#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

__kernel void bitonic(__global float *as, const unsigned k, const unsigned n) {
    const unsigned int gid = get_global_id(0);

    unsigned int idx_i = gid / (k / 2);
    unsigned int idx_j = (gid - idx_i * k / 2) / n;
    unsigned int start = gid + idx_i * k / 2 + idx_j * n;

    if (as[start] >= as[start + n]) {
        if (idx_i % 2 == 0) {
            float temp = as[start];
            as[start] = as[start + n];
            as[start + n] = temp;
        }

    } else if (idx_i % 2 == 1) {
        float temp = as[start];
        as[start] = as[start + n];
        as[start + n] = temp;
        }
    }



