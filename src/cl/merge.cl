__kernel void merge(__global float *as, __global float *as_, unsigned int n, unsigned int level) {
    unsigned int i = get_global_id(0);

    unsigned int left = ((i >> level) << level);
    unsigned int right = left + (1 << level);
    unsigned int mid = (left + right) / 2;

    if (i >= n || mid >= n) {
        as_[i] = as[i];
        return;
    }
    if (right > n)
        right = n;
    
    int l = -1, r = i >= mid ? mid - left : right - mid;
    int j_offset = i < mid ? mid : left;
    int i_loc = i - (i < mid ? left : mid);
    while (r - l > 1) {
        int j = (l + r) / 2;
        int p = j_offset + j;
        if (as[j + j_offset] < as[i] || (as[j + j_offset] == as[i] && i < mid))
            l = j;
        else
            r = j;
    }

    //as_[i] = as[left];
    as_[left + i_loc + r] = as[i];
}