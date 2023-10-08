#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_GROUP_SIZE 256

unsigned int bin_search(__global const float* as, unsigned int left, unsigned int right, float value, bool is_left){
    unsigned int middle = (left + right) / 2;
    while (right > left){
        if ((is_left && (as[middle] == value)) || as[middle] > value) {
            right = middle;
        } else {
            left = middle + 1;
        }
        middle = (left + right) / 2;
    }
    return left;
}

__kernel void merge(__global float* as, __global float* bs, unsigned  int k, unsigned  int n) {
    const unsigned  int gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    float value = as[gid];
    const unsigned  int start  = gid - gid % (2 * k);
    const unsigned  int end = (start + 2 * k <= n) ? start + 2 * k : n;
    const unsigned  int middle = start + k;

    unsigned  int new_index = 0;
    if (gid < middle) {
        unsigned  int left_shift = gid - start;
        new_index = left_shift + bin_search(as, middle, end, value, true) - k;
    } else {
        unsigned int right_shift = gid - middle;
        new_index = right_shift + bin_search(as, start, middle, value, false);
    }
    bs[new_index] = value;
}

