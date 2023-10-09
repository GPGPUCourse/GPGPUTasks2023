
__kernel void merge(__global const float *src, __global float *dest, const unsigned int cur_n) {
    const unsigned int gid = get_global_id(0);

    // we ain't even interested in local id

    const unsigned int halfn = cur_n / 2; // let's assume that the length is a power of two. If data doesn't match - we'll make it by filling infs.

    const unsigned int kgid = gid % cur_n;

    const unsigned int start_ind = (gid / cur_n) * cur_n;

    const bool from_right = kgid >= halfn; // for example n = 4, halfn = 2; {0, 1} - from left, {2, 3} - from right

    const float val = src[gid];
    const unsigned int base_ind = start_ind + (!from_right) * halfn;

    int l = -1;
    int r = halfn;

    while (l < (r - 1)) {
        const unsigned int ind = (l + r) / 2;
        const float mid_val = src[base_ind + ind];
        const bool bigger = from_right ? val >= mid_val : val > mid_val;
        l = bigger ? ind : l;
        r = !bigger ? ind : r;
    }

    dest[start_ind + (kgid % halfn) + r] = val;
}