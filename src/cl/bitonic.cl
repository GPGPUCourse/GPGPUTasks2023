#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

__kernel void bitonic(__global float *as, int n, int global_width, int width) {
    int gid = get_global_id(0);
    int step = width / 2;

    int block = 2 * gid / width;
    int dir = (2 * gid / global_width) % 2;

    int i = block * width + gid % step;
    int j = i + step;
    float asi = as[i];
    float asj = as[j];
    if (j < n) {
        if (dir ? asi < asj : asi > asj) {
            as[i] = asj;
            as[j] = asi;
        }
    }
}
