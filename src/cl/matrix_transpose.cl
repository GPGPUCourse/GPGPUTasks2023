typedef unsigned int _uint;
#define SIZE 16
__kernel void matrix_transpose(__global const float *src, __global float *res, const _uint size1, const _uint size0) {
    _uint lid0 = get_local_id(0);
    _uint lid1 = get_local_id(1);

    _uint gid0 = get_global_id(0);
    _uint gid1 = get_global_id(1);

    __local float tile[SIZE][SIZE];

    if (gid0 < size0 && gid1 < size1) {
        tile[lid0][lid1] = src[size0 * gid1 + gid0];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    _uint ls0 = get_local_size(0);
    _uint ls1 = get_local_size(1);

    _uint ggid0 = get_group_id(0);
    _uint ggid1 = get_group_id(1);

    gid0 = ls1 * ggid1 + lid0;
    gid1 = ls0 * ggid0 + lid1;
    if (gid0 < size1 && gid1 < size0) {
        res[size1 * gid1 + gid0] = tile[lid1][lid0];
    }
}