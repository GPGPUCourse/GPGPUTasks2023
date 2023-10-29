#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#include <float.h>
#endif

#define get_masked(x, shift, mask) (((x) >> (shift)) & (mask))

__kernel void merge_sort(__global const unsigned int *as,
                        __global unsigned int *bs,
                        const unsigned int n,
                        const unsigned int width,
                        const unsigned int shift,
                        const unsigned int mask)
{
    int globalId = get_global_id(0);
    int groupId = globalId / width;

    int leftA = groupId * width;
    int rightA = leftA + width;
    int localId = globalId - leftA;

    bool isRight = groupId % 2;

    int otherL = isRight ? leftA - width : rightA;
    int otherR = isRight ? leftA : rightA + width;
    int start = isRight ? leftA - width : leftA;

    int l = otherL - 1, r = otherR, m = 0;

    const unsigned int uint_max = (1L << sizeof(unsigned int) * 8) - 1;
    const unsigned int uint_min = 0;
    bool isLeft = !isRight;
    unsigned int v = globalId < n ? as[globalId] : uint_max;
    unsigned int masked_v = get_masked(v, shift, mask);
    unsigned int base = 0;

    while (r - l > 1) {
        m = (l + r) / 2;
        base = get_masked(as[m], shift, mask);

        if (m >= otherR){
            base = uint_max;
        } else if (m < otherL) {
            base = uint_min;
        }

        if (masked_v < base || isLeft && masked_v <= base)
            r = m;
        else
            l = m;
    }

    unsigned int index = start + localId + r - otherL;

    if (globalId < n && index < n){
        bs[index] = v;
    }
}

__kernel void merge_sort_float(__global const float *as,
                        __global float *bs,
                        const unsigned int n,
                        const unsigned int width) {
    int globalId = get_global_id(0);
    int groupId = globalId / width;

    int leftA = groupId * width;
    int rightA = leftA + width;
    int localId = globalId - leftA;

    bool isRight = groupId % 2;

    int otherL = isRight ? leftA - width : rightA;
    int otherR = isRight ? leftA : rightA + width;
    int start = isRight ? leftA - width : leftA;

    int l = otherL - 1, r = otherR, m = 0;

    bool isLeft = !isRight;
    float v = globalId < n ? as[globalId] : FLT_MAX;
    float base = FLT_MIN;

    while (r - l > 1) {
        m = (l + r) / 2;
        base = as[m];

        if (m >= otherR){
            base = FLT_MAX;
        } else if (m < otherL) {
            base = FLT_MIN;
        }

        if (v < base || isLeft && v <= base)
            r = m;
        else
            l = m;
    }

    unsigned int index = start + localId + r - otherL;

    if (globalId < n && index < n){
        bs[index] = v;
    }
}

#undef get_masked
