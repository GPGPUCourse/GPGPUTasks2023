#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#include <float.h>
#endif

__kernel void merge(__global const float *as,
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
