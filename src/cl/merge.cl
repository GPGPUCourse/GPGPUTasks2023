#ifdef __CLION_IDE__

#include <cl/clion_defines.cl>
#include <math.h>

#endif

#line 6

__kernel void merge(const __global float *as_gpu, __global float* bs_gpu, const uint stride, const uint n) {
    const uint baseIndex = get_global_id(0);
    if (baseIndex >= n) return;

    const uint group = baseIndex / stride;
    const uint paired = (group % 2 == 0) ? group + 1 : group - 1;

    int l = paired * stride;
    int r = min(l + stride, n);
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (as_gpu[mid] > as_gpu[baseIndex] || ((group % 2 == 0) && as_gpu[mid] == as_gpu[baseIndex]))
            r = mid;
        else
            l = mid + 1;
    }
    const uint i = min(group, paired) * stride + (baseIndex % stride) + (r - paired * stride);
    bs_gpu[i] = as_gpu[baseIndex];
}

