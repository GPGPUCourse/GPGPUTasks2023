__kernel void prefix_sum(const __global unsigned *as, __global unsigned *bs, const unsigned offset,
                         const unsigned size) {
    unsigned gid = get_global_id(0);

    if (gid >= size) {
        return;
    }

    if (gid >= offset) {
        bs[gid] = as[gid] + as[gid - offset];
    } else {
        bs[gid] = as[gid];
    }
}