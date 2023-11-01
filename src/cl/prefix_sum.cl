#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

int pow(int base, int exponent) {
    int result = 1;
    for (int i = 0; i < exponent; ++i) {
        result *= base;
    }
    return result;
}


__kernel void prefix_sum_1_up_sweep(__global unsigned int *as,
                    const unsigned int d,
                    const unsigned int n) {
    const unsigned int gid = get_global_id(0);
    unsigned int idx = gid * d + d - 1;

    if (idx >= n)
        return;

    as[idx] += as[idx - d / 2];
}


__kernel void prefix_sum_2_down_sweep(__global unsigned int *res,
                    __global unsigned int *as,
                    const unsigned int n) {

    unsigned int gid = get_global_id(0);
    unsigned int idx = gid + 1;

    if (gid >= n)
        return;

    int index_0 = 0;
    unsigned int temp = idx;

    int max_index = 0;

    while (temp > 0) {
        if (temp & 1)
            max_index += pow(2, index_0);
        temp >>= 1;
        index_0++;
    }

    res[gid] += as[max_index - 1];
    int index_1 = 0;

    while (idx > 0) {
        if (idx & 1) {
            max_index -= pow(2, index_1);
            if (max_index > 0)
                res[gid] += as[max_index - 1];
        }
        idx >>= 1;
        index_1++;
    }
}