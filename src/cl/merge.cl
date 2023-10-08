#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif


unsigned int binary_search(bool equal, const float x, unsigned int left,
                           unsigned int right, __global const float *a) {
    while (left < right) {

        unsigned int mid = (left + right) / 2;

        if ((a[mid] <= x && equal) || (a[mid] < x && !equal))
            left = mid + 1;
        else
            right = mid;
    }
    return left;
}


__kernel void merge(__global const float *src, __global float *res,
                    const unsigned curr_size, const unsigned n) {
    const unsigned int gid = get_global_id(0);

    if (gid >= n)
        return;

    unsigned int left = (gid / curr_size) * curr_size;
    const unsigned int right = left + curr_size;
    const unsigned int mid = (left + right) / 2;
    const unsigned int start = left + gid % curr_size;

    const float x = src[gid];
    unsigned int position;

    if (gid < mid) {
        const unsigned int bias = binary_search(false, x, mid, right, src) - mid;
        position = start + bias;
    }
    else {
        const unsigned int bias = binary_search(true, x, left, mid, src) - left;
        position = start - curr_size / 2 + bias;
    }
    res[position] = x;
}
