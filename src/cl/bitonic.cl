#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define max(x, y) x > y ? x : y
#define min(x, y) x < y ? x : y

__kernel void bitonic(__global float *as, unsigned int chunk, unsigned int stride, unsigned int n) {
    unsigned idx = get_global_id(0);
    unsigned id2x = idx << 1;

    if (id2x >= n) return;
    
    unsigned s = id2x / chunk;
    idx = s * chunk
        + (idx - idx / stride * stride) 
        + (id2x - s * chunk ) / (stride << 1) * (stride << 1);

    float max = max(as[idx], as[idx + stride]);
    float min = min(as[idx], as[idx + stride]);

    as[idx] = (s & 1) ? max : min;
    as[idx + stride] = (s & 1) ? min : max;
}
