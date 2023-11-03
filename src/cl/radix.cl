#define _uint unsigned int
#define CHECK_BIT(var, pos) ((var) & (1 << (pos)))
#define MAX_NUM_OFSIZE(n) ((1 << (n + 1)) - 1)
#define BITWISE_AND(lhs, rhs) ((lhs) & (rhs))

#define NUM_BITS 4
#define POWER_OF_BITS (1 << NUM_BITS)
#define TILE_ROWS 16
#define TILE_COLS (TILE_ROWS + 1)
#define MAX_NUM (POWER_OF_BITS - 1)
#define GET_PART(num, iternum) ((num >> (iternum * NUM_BITS)) & MAX_NUM)
#define WG_SIZE 128

__kernel void reset(__global _uint *as) {
    int gid = get_global_id(0);
    if (gid < get_global_size(0)) {
        as[gid] = 0;
    }
}

__kernel void count(__global _uint *as, __global _uint *cnts, _uint iternum) {
    _uint gid = get_global_id(0);
    _uint lid = get_local_id(0);
    _uint wid = get_group_id(0);

    __local _uint local_cnt[POWER_OF_BITS];

    // someone will succeed in writing zero. no code divergence!
    local_cnt[lid % POWER_OF_BITS] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    _uint num = as[gid];

    _uint part = GET_PART(num, iternum);

    atomic_inc(&local_cnt[part]);
    barrier(CLK_LOCAL_MEM_FENCE);

    // no way to evade code divergence here... sad.
    if (!(lid < POWER_OF_BITS))
        return;

    cnts[(POWER_OF_BITS * wid) + lid] = local_cnt[lid];
}

__kernel void reorder(__global _uint *as, __global _uint *bs, __global _uint *psum_cnts, __global _uint *cnts, _uint wgscnt, _uint iternum) {
    _uint gid = get_global_id(0);
    _uint lid = get_local_id(0);
    _uint wid = get_group_id(0);

    // let's save our group's as, as we need them more than once
    __local _uint local_as[WG_SIZE];
    local_as[lid] = as[gid];

    // let's save our group's prefix sums
    __local _uint local_psum_cnt[POWER_OF_BITS];
    if (lid < POWER_OF_BITS)
        local_psum_cnt[lid] = psum_cnts[wgscnt * lid + wid];

    barrier(CLK_LOCAL_MEM_FENCE);

    _uint num = local_as[lid];
    _uint part = GET_PART(num, iternum);// this is our current digit that we're sorting by

    _uint loc_res_ind = 0;// this tells us how many equal digits are at positions less than ours
    for (_uint i = 0; i < WG_SIZE; i++) {
        bool iless = i < lid;

        _uint ipart = GET_PART(local_as[i], iternum);
        bool iparteq = ipart == part;

        loc_res_ind += (iless && iparteq) ? 1 : 0;
    }

    _uint res_index = loc_res_ind + local_psum_cnt[part] - cnts[wid * POWER_OF_BITS + part];
    bs[res_index] = num;// finally
}

__kernel void psum_naive(__global _uint *reduce_lst, const int power) {
    const int gid = get_global_id(0);

    const int ind = (gid / power) * 2 * power + (gid % power) + power;
    reduce_lst[ind] += reduce_lst[ind / power * power - 1];
}

__kernel void prefix_sum_reduce(__global _uint *reduce_lst, const int power) {
    const int gid = get_global_id(0);
    const int n = get_global_size(0);

    const int chunk_size = 1 << power;
    const int power_zero = power == 0;
    const int rhs = power_zero ? gid : gid + (1 << (power - 1));
    const bool addition = rhs < n && gid % chunk_size == 0;
    reduce_lst[gid] = power_zero ? reduce_lst[gid] : (addition ? reduce_lst[gid] + reduce_lst[rhs] : reduce_lst[gid]);
}

__kernel void prefix_sum_write(__global _uint *reduce_lst, __global _uint *result, const int power) {
    const int gid = get_global_id(0);
    const int n = get_global_size(0);

    const bool need_this_power = CHECK_BIT(gid, power);
    const int max_num = MAX_NUM_OFSIZE(power);
    const int needed_ind = gid - BITWISE_AND(max_num, gid);
    result[gid] += need_this_power ? reduce_lst[needed_ind] : 0;
}

__kernel void matrix_transpose(__global const _uint *mat, __global _uint *mat_tr, int nrow, int ncol) {
    const int col = get_global_id(0);
    const int row = get_global_id(1);

    if (row >= nrow || col >= ncol)
        return;

    mat_tr[(col * nrow) + row] = mat[(row * ncol) + col];
}
