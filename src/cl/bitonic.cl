#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

__kernel void bitonic(__global float *as, int K, int local_k) {
    int gid = get_global_id(0);
    int r_id = gid / local_k;
    int rl_id = gid - r_id * local_k;
    int idx = 2 * local_k * r_id + rl_id;
    bool is_left = (gid / K) % 2 == 0 ? as[idx + local_k] < as[idx] : as[idx + local_k] > as[idx];
    if (is_left) {
        float tmp = as[idx];
        as[idx] = as[idx + local_k];
        as[idx + local_k] = tmp;
    }
}
