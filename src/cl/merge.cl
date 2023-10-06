#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

inline bool comp(float e1, float e2, bool weak) {
    return weak ? e1 < e2 : e1 <= e2;
}

int binary_search(const __global float *a, int n, float e, bool weak) {
    int lo = -1;
    int hi = n;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (comp(a[mid], e, weak)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo + 1;
}

__kernel void merge(const __global float *in, __global float *out, const int sorted) {
    const int i = get_global_id(0);
    const int block_begin = (i / (2 * sorted)) * (2 * sorted);
    bool at_left = (i % (2 * sorted)) < sorted;
    const int l = block_begin + at_left * sorted;
    const int less = binary_search(in + l, sorted, in[i], !at_left);
    out[i - sorted * !at_left + less] = in[i];
}
