#line 6

#define uint32_t unsigned int
#define MASK_WIDTH 4
#define CNT (1 << MASK_WIDTH)
#define TILE_SIZE 16
#define WG_SIZE 128

__kernel void fill_zeros(__global uint32_t* as,
                         const uint32_t N)
{
    const uint32_t gid = get_global_id(0);
    if (gid >= N)
        return;
    as[gid] = 0;
}

__kernel void local_stable_merge_sort(__global const uint32_t* as,
                                __global uint32_t* bs,
                                const uint32_t mask,
                                const uint32_t N,
                                const uint32_t M)
{
    const uint32_t gid = get_global_id(0);
    const uint32_t curr = as[gid];
    const uint32_t curr_masked = curr & mask;

    uint32_t other;
    uint32_t other_masked;

    const uint32_t group_id = gid / M;
    const uint32_t left_bound = group_id * M;
    const uint32_t right_bound = left_bound + M;
    const uint32_t local_id = gid - left_bound;

    bool is_right = group_id % 2;

    uint32_t L = is_right ? left_bound - M : right_bound;
    uint32_t R = is_right ? left_bound : L + M;
    uint32_t med = 0;

    while (L < R) {
        med = (L + R) / 2;
        other = as[med];
        other_masked = other & mask;
        if ((!is_right && curr_masked <= other_masked) || curr_masked < other_masked) {
            R = med;
        } else {
            L = med + 1;
        }
    }

    const uint32_t index = local_id + R - !is_right * M;
    if (index < N) {
        bs[index] = curr;
    }
}

__kernel void local_counts(__global const uint32_t* as,
                           const uint32_t N,
                           __global uint32_t* matrix_local_cnt,
                           const uint32_t mask,
                           const uint32_t mask_offset)
{
    const uint32_t gid = get_global_id(0);
    const uint32_t lid = get_local_id(0);
    const uint32_t group_id = get_group_id(0);

    __local uint32_t local_mem[CNT];
    if (lid < CNT) {
        local_mem[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(&local_mem[(as[gid] & mask) >> mask_offset], 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < CNT) {
        matrix_local_cnt[group_id * CNT + lid] = local_mem[lid];
    }
}

__kernel void local_counts_transposed(__global const uint32_t* as,
                           const uint32_t N,
                           __global uint32_t* bs,
                           const uint32_t mask,
                           const uint32_t mask_offset)
{
    const uint32_t gid = get_global_id(0);
    const uint32_t lid = get_local_id(0);
    const uint32_t group_id = get_group_id(0);
    const uint32_t groups_cnt = get_num_groups(0);

    __local uint32_t local_mem[CNT];
    if (lid < CNT) {
        local_mem[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(&local_mem[(as[gid] & mask) >> mask_offset], 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid < CNT) {
        bs[groups_cnt * lid + group_id] = local_mem[lid];
    }
}

__kernel void prefix_sum(__global const uint32_t* as,
                         __global uint32_t* bs,
                         const uint32_t N,
                         const uint32_t offset)
{
    const uint32_t gid = get_global_id(0);

    if (gid >= N)
        return;

    bs[gid] = gid >= offset ? as[gid] + as[gid - offset] : as[gid];
}

__kernel void shift_right(__global const uint32_t* as,
                          __global uint32_t* bs,
                          const uint32_t N)
{
    const uint32_t gid = get_global_id(0);

    if (gid + 1 >= N)
        return;

    bs[gid + 1] = as[gid];

    if (gid == 0)
        bs[gid] = 0;
}

__kernel void matrix_transpose(__global const uint32_t* from,
                               __global uint32_t* to,
                               const uint32_t M,
                               const uint32_t K)
{
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    const unsigned int gr_i = get_group_id(0);
    const unsigned int gr_j = get_group_id(1);

    const unsigned int ls_i = get_local_size(0);
    const unsigned int ls_j = get_local_size(1);

    __local float tile[TILE_SIZE][TILE_SIZE * 2];

    if (i < K && j < M) {
        tile[local_i][local_i + local_j] = from[j * K + i];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    const uint32_t ni = ls_j * gr_j + local_i;
    const uint32_t nj = ls_i * gr_i + local_j;

    if (ni < M && nj < K) {
        to[M * nj + ni] = tile[local_j][local_i + local_j];
    }
}

__kernel void radix(__global const uint32_t* as,
                    __global uint32_t* bs,
                    const uint32_t N,
                    __global const uint32_t* local_prefix_sums,
                    __global const uint32_t* global_prefix_sums,
                    const uint32_t wgCnt,
                    const uint32_t mask,
                    const uint32_t mask_offset)
{
    const uint32_t gid = get_global_id(0);
    const uint32_t lid = get_local_id(0);
    const uint32_t group_id = get_group_id(0);

    if (gid >= N)
        return;

    const uint32_t curr_masked_offset = (as[gid] & mask) >> mask_offset;
    const uint32_t curr_offset = local_prefix_sums[group_id * CNT + curr_masked_offset] - local_prefix_sums[group_id * CNT];
    const uint32_t index = global_prefix_sums[wgCnt * curr_masked_offset + group_id] + lid - curr_offset;

    bs[index] = as[gid];
}