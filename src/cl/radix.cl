#define WORK_GROUP_SIZE 64
#define COUNT_NUMBER 8
#define WORK_PER_THREAD 8  // = WORK_GROUP_SIZE / COUNT_NUMBER
#define BIT_PER_LEVEL 3  // = log_2(COUNT_NUMBER)


__kernel void count(__global unsigned int *as, __global unsigned int *counts, unsigned int level, unsigned int n) {
    int i_g = get_global_id(0), i_l = get_local_id(0);
    int group_number = get_group_id(0);
    __local int local_counts[COUNT_NUMBER][WORK_GROUP_SIZE / COUNT_NUMBER];

    local_counts[i_l % COUNT_NUMBER][i_l] = 0;

    if (i_g < n) {
        local_counts[i_l % COUNT_NUMBER][i_l / COUNT_NUMBER] = 0;
        for (int i = 0; i < WORK_PER_THREAD; ++i) {
            unsigned int value = as[i_g - i_l + i_l / COUNT_NUMBER * COUNT_NUMBER + i];
            value = (value >> (level * BIT_PER_LEVEL)) & (COUNT_NUMBER - 1);
            local_counts[i_l % COUNT_NUMBER][i_l / COUNT_NUMBER] += (int)(value == (i_l % COUNT_NUMBER));
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        if (i_l / COUNT_NUMBER == 0) {
            for (int i = 1; i < WORK_GROUP_SIZE / WORK_PER_THREAD; ++i) {
                local_counts[i_l % COUNT_NUMBER][0] += local_counts[i_l % COUNT_NUMBER][i];
            }
            counts[group_number * COUNT_NUMBER + i_l % COUNT_NUMBER] = local_counts[i_l % COUNT_NUMBER][0];
        }
    }
}


__kernel void radix(__global unsigned int *as, __global unsigned int *res, __global unsigned int *counts, unsigned int level, unsigned int n) {
    int i_g = get_global_id(0), i_l = get_local_id(0);
    int wg_total = get_num_groups(0), wg_id = get_group_id(0);

    if (i_g < n) {
        int value = (as[i_g] >> (level * BIT_PER_LEVEL)) & (COUNT_NUMBER - 1);
        unsigned int pre_count = 0;
        for (int i = 0; i < i_l; ++i) {
            pre_count += (int)(as[i_g - i_l + i] == value);
        }

        unsigned int index = pre_count + counts[wg_total * value + wg_id];
        res[index] = as[i_g];
        // res[i_g] = index;
    }
}
