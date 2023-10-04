__kernel void merge_sort(__global float* as, __global float* bs, int k, unsigned int n) {
    int i = get_global_id(0), i0, i1, new_index;
    int lo, hi, mid;
    float tmp;
    
    i1 = i % (2 * k);
    i0 = i - i1;

    if (i < n) {
        tmp = as[i];
        if (i1 < k) {
            lo = i0 + k, hi = i0 + 2 * k;
            while (lo < hi) {
                mid = (lo + hi) / 2;
                if (mid >= n || as[mid] >= tmp) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            new_index = i1 + lo - k;
        }
        else {
            lo = i0, hi = i0 + k;
            while (lo < hi) {
                mid = (lo + hi) / 2;
                if (as[mid] > tmp) {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            new_index = i1 + lo - k;
        }
    }

    if (i < n) {
        bs[new_index] = tmp;
    }
}
