__kernel void fill_zero(__global unsigned *as, const unsigned size) {
    unsigned gid = get_global_id(0);

    if (gid < size) {
        as[gid] = 0;
    }
}

__kernel void shift(const __global unsigned *as, __global unsigned *bs, const unsigned size) {
    int gid = get_global_id(0);

    if (gid >= size) {
        return;
    }

    bs[gid] = gid > 0 ? as[gid - 1] : 0;
}