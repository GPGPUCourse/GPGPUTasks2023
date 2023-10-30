#define NUM_BITS 4
#define POWER_OF_BITS (1 << NUM_BITS)
#define MAX_NUM (POWER_OF_BITS - 1)
#define GET_PART(num, iternum) ((num >> (iternum * NUM_BITS)) & MAX_NUM)
#define WG_SIZE 128

__kernel void count_local(__global unsigned int *as, __global unsigned int *cnts, unsigned int iternum) {
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int wid = get_group_id(0);

    __local unsigned int local_cnt[POWER_OF_BITS];

    // someone will succeed in writing zero. no code divergence!
    local_cnt[lid % POWER_OF_BITS] = 0;

    unsigned int num = as[gid];

    unsigned int part = GET_PART(num, iternum);

    atomic_inc(&local_cnt[part]);

    barrier(CLK_LOCAL_MEM_FENCE);

    // no way to evade code divergence here... sad.
    if (!(lid < POWER_OF_BITS))
        return;

    atomic_add(&cnts[(POWER_OF_BITS * wid) + lid], local_cnt[lid]);
}

__kernel void reorder(__global unsigned int *as, __global unsigned int *bs, __global unsigned int *cnts,
                      __global unsigned int *psum_cnts, unsigned int iternum) {
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);
    unsigned int wid = get_group_id(0);

    // let's save our group's as, as we need them more than once
    __local unsigned int local_as[WG_SIZE];
    local_as[lid] = as[gid];

    // let's save our group's prefix sums
    __local unsigned int local_psum_cnt[POWER_OF_BITS];
    if (lid < POWER_OF_BITS)
        local_psum_cnt[lid] = psum_cnts[(POWER_OF_BITS * wid) + lid];

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int num = local_as[lid];
    unsigned int part = GET_PART(num, iternum);// this is our current digit that we're sorting by

    unsigned int loc_res_ind = 0;// this tells us how many equal digits are at positions less than ours
    for (unsigned int i = 0; i < WG_SIZE; i++) {
        bool iless = i < lid;

        unsigned int ipart = GET_PART(local_as[i], iternum);
        bool iparteq = ipart == part;

        loc_res_ind += (iless && iparteq) ? 1 : 0;
    }

    unsigned int res_index = loc_res_ind + local_psum_cnt[part];

    bs[res_index] = num;// finally
}
