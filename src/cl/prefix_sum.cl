#define _uint unsigned int
#define CHECK_BIT(var, pos) ((var) & (1 << (pos)))
#define MAX_NUM_OFSIZE(n) ((1 << (n + 1)) - 1)
#define BITWISE_AND(lhs, rhs) ((lhs) & (rhs))

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

    const int gid_from_one = gid + 1;
    const bool need_this_power = CHECK_BIT(gid_from_one, power);
    const int max_num = MAX_NUM_OFSIZE(power);
    const int needed_ind = gid_from_one - BITWISE_AND(max_num, gid_from_one);
    result[gid] += need_this_power ? reduce_lst[needed_ind] : 0;
}
