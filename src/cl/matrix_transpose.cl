#define SIZE 32
__kernel void transpose(__global const unsigned *src, __global unsigned *res, const unsigned size1, const unsigned size0) {
    unsigned lid0 = get_local_id(0);
    unsigned lid1 = get_local_id(1);

    unsigned gid0 = get_global_id(0);
    unsigned gid1 = get_global_id(1);

    __local unsigned tile[SIZE / 2][SIZE];

    if (gid0 < size0 && gid1 < size1) {
        tile[lid0][lid1 + lid0] = src[size0 * gid1 + gid0];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned ls0 = get_local_size(0);
    unsigned ls1 = get_local_size(1);

    unsigned ggid0 = get_group_id(0);
    unsigned ggid1 = get_group_id(1);

    gid0 = ls1 * ggid1 + lid0;
    gid1 = ls0 * ggid0 + lid1;
    if (gid0 < size1 && gid1 < size0) {
        res[size1 * gid1 + gid0] = tile[lid1][lid0 + lid1];
    }
}
