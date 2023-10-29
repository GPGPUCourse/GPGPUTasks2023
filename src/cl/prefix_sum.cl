__kernel void prefix_sum(__global unsigned *as, __global unsigned *bs, unsigned n, unsigned blockSize) {
    unsigned idx = get_global_id(0);
    if (idx >= n)
        return;
    if (idx < blockSize)
        bs[idx] = as[idx];
    else
        bs[idx] = as[idx] + as[idx - blockSize];
}