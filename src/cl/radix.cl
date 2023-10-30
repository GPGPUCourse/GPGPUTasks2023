#define WORK_GROUP_SIZE 64
#define COUNT_NUMBER 8
#define WORK_PER_THREAD 8  // = WORK_GROUP_SIZE / COUNT_NUMBER
#define BIT_PER_LEVEL 3  // = log_2(COUNT_NUMBER)


__kernel void count(__global unsigned int *as, __global unsigned int *counts, unsigned int level, unsigned int n) {
    unsigned int i_g = get_global_id(0), i_l = get_local_id(0);
    unsigned int group_number = get_group_id(0);
    unsigned int shift1 = level * BIT_PER_LEVEL * (int)(level * BIT_PER_LEVEL < 32) + 31 * (int)(level * BIT_PER_LEVEL >= 32);
    unsigned int shift2 = level * BIT_PER_LEVEL - shift1;
    __local unsigned int local_counts[COUNT_NUMBER][WORK_GROUP_SIZE / COUNT_NUMBER];

    if (i_g < n) {
        local_counts[i_l % COUNT_NUMBER][i_l / COUNT_NUMBER] = 0;
        for (unsigned int i = 0; i < COUNT_NUMBER; ++i) {
            unsigned int value = as[i_g - i_l % COUNT_NUMBER + i];
            value = ((((value >> shift1) & 63) >> shift2) & (unsigned int)(COUNT_NUMBER - 1));
            local_counts[i_l % COUNT_NUMBER][i_l / COUNT_NUMBER] += (int)(value == (i_l % COUNT_NUMBER));
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (i_l / COUNT_NUMBER == 0) {
            for (int i = 1; i < COUNT_NUMBER; ++i) {
                local_counts[i_l % COUNT_NUMBER][0] += local_counts[i_l % COUNT_NUMBER][i];
            }
            counts[group_number * COUNT_NUMBER + i_l % COUNT_NUMBER] = local_counts[i_l % COUNT_NUMBER][0];
        }
    }
}


__kernel void radix(__global unsigned int *as, __global unsigned int *res, __global unsigned int *counts, unsigned int level, unsigned int n) {
    int i_g = get_global_id(0), i_l = get_local_id(0);
    int wg_total = get_num_groups(0), wg_id = get_group_id(0);
    unsigned int shift1 = level * BIT_PER_LEVEL * (int)(level * BIT_PER_LEVEL < 32) + 31 * (int)(level * BIT_PER_LEVEL >= 32);
    unsigned int shift2 = level * BIT_PER_LEVEL - shift1;

    if (i_g < n) {
        unsigned int value = ((((as[i_g] >> shift1) & 63) >> shift2) & (unsigned int)(COUNT_NUMBER - 1));
        unsigned int pre_count = 0;
        for (int i = 0; i < i_l; ++i) {
            pre_count += (int)(((((as[i_g - i_l + i] >> shift1) & 63) >> shift2) & (unsigned int)(COUNT_NUMBER - 1)) == value);
        }

        unsigned int index = pre_count + (value == 0 && wg_id == 0 ? 0 : counts[wg_total * value + wg_id - 1]);
        res[index] = as[i_g];
        // res[i_g] = index;
    }
}
