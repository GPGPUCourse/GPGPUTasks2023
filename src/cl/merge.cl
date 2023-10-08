
__kernel void merge(__global float *a, const unsigned int n) {
    const unsigned int gid = get_global_id(0);

    // we ain't even interested in local id

    const unsigned int halfn = n >> 1; // let's assume that the length is a power of two. If data doesn't match - we'll make it by filling infs.

    const bool from_right = gid >= halfn; // for example n = 4, halfn = 2; {0, 1} - from left, {2, 3} - from right

    const float val = a[gid];
    const unsigned int base_ind = (!from_right) * halfn;

    unsigned int l = -1;
    unsigned int r = halfn;

    unsigned int ind = -100; // This is so if something goes wrong - let it go terribly wrong :D

    while (l < r - 1) {
        ind = (l + r) >> 1;
        const float mid_val = a[base_ind + ind];
        const bool bigger = val > mid_val;
        l = bigger ? ind : l;
        r = bigger ? r : ind;
    }

    // wait for everyone to find their ind
    barrier(CLK_GLOBAL_MEM_FENCE);

    a[(gid % halfn) + r] = val;
}