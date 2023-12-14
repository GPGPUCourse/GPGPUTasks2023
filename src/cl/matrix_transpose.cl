#define M 16
#define K 16

__kernel void matrix_transpose(__global float* as, __global float* as_t, const uint m, const uint k) {
    const uint gi = get_global_id(0);
    const uint gj = get_global_id(1);
    const uint li = get_local_id(0);
    const uint lj = get_local_id(1);

    __local float tile[M * K];
    tile[lj * M + li] = as[gj * m + gi];
    barrier(CLK_LOCAL_MEM_FENCE);

    const uint gi_t = gi - li + lj;
    const uint gj_t = gj - lj + li;
    as_t[gi_t * m + gj_t] = tile[li * K + lj];
}