__kernel void prefix_sum_naive(__global unsigned *as, __global unsigned *bs, const unsigned offset,
                               const unsigned size) {
    unsigned gid = get_global_id(0);

    if (gid >= offset) {
        bs[gid] = as[gid] + as[gid - offset];
    } else {
        bs[gid] = as[gid];
    }
}

__kernel void prefix_sum_up_sweep(__global unsigned *as, __global unsigned *bs, const unsigned offset,
                                  const unsigned size) {
    unsigned gid = get_global_id(0);

    if (gid >= size)
        return;

    if (!((gid + 1) % (offset << 1))) {
        bs[gid] = as[gid] + as[gid - offset];
    } else {
        bs[gid] = as[gid];
    }
}

__kernel void prefix_sum_down_sweep(__global unsigned *as, __global unsigned *bs, const unsigned offset,
                                    const unsigned size) {
    unsigned gid = get_global_id(0);

    if (gid >= size)
        return;

    if (!((gid + 1) % (offset << 1))) {
        bs[gid - offset] = as[gid];
        bs[gid] = as[gid] + as[gid - offset];
    } else if ((1 + gid) % offset) {
        bs[gid] = as[gid];
    }
}

__kernel void prefix_sum_shift(__global unsigned *as, __global unsigned *bs, const unsigned total, const unsigned size) {
    unsigned gid = get_global_id(0), ngid = gid + 1;
    if (ngid > size)
        return;
    
    bs[gid] = (ngid == size) ? total : as[ngid] - total;
}