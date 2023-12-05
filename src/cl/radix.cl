__kernel void prefix_sum(__global const unsigned int *as, __global unsigned int *bs, unsigned int n, unsigned int block_size) {
    int id = get_global_id(0);
    if (id < n && ((id + 1) & block_size)){
        if (id >= 262144 || id < 0)
            printf("Sum bs: %d \n", id);
        if (((id + 1) / block_size) - 1 >= 262144 || ((id + 1) / block_size) - 1 < 0)
            printf("Sum as: %d \n", ((id + 1) / block_size) - 1);
        bs[id] += as[((id + 1) / block_size) - 1];
    }
}

__kernel void reduce(__global const unsigned int *as, __global unsigned int *res, unsigned int n) {
    int id = get_global_id(0);
    if (id < n) {
        if (id >= 262144 || id < 0)
            printf("Reduce res: %d \n", id);
        if (id * 2 >= 262144 || id * 2 < 0)
            printf("Reduce as1: %d \n", id);
        if (id * 2 + 1 >= 262144 || id * 2  + 1 < 0)
            printf("Reduce as2: %d \n", id);
        res[id] = as[id * 2] + as[id * 2 + 1];
    }
}

__kernel void cnt_calc(__global const unsigned int *as, __global unsigned int *counters, unsigned int current_bit, unsigned int n) {
    int id = get_global_id(0);
    if (id < n) {
        if (id >= 262144 || id < 0)
            printf("Counter id1: %d \n", id);
        counters[id] = 0;
        int start = id * 128;
        for (int i = start; i < start + 128; i++) {
            if (i >= 33554432 || i < 0)
                printf("Counter i: %d \n", i);
            if (as[i] & (1 << current_bit)) {
                if (id >= 262144 || id < 0)
                    printf("Counter id2: %d \n", id);
                counters[id]++;

            }
        }
    }
}

void sort_inside_block(__local unsigned int *part, __local unsigned int *number_of_ones, unsigned int current_bit) {
    unsigned int part_copy[128];
    for (unsigned int i = 0; i < 128; ++i)
        part_copy[i] = part[i];

    unsigned int current_one = 128 - *number_of_ones;
    unsigned int current_zero = 0;

    for (unsigned int i = 0; i < 128; ++i)
        if (part_copy[i] & (1 << current_bit)){
            if (current_one >= 128 || current_one < 0)
                printf("Sort inside part(128) index1: %d \n", current_one);
            part[current_one++] = part_copy[i];
        } else {
            if (current_zero >= 128 || current_zero < 0)
                printf("Sort inside part(128) index2: %d \n", current_zero);
            part[current_zero++] = part_copy[i];
        }
}

__kernel void radix(__global const unsigned int *prefix_cnt, __global unsigned int *as, __global unsigned int *res,
                        unsigned int current_bit, unsigned int n) {
    const unsigned int id = get_global_id(0);
    const unsigned int block_id = get_group_id(0);
    const unsigned int id_inside_block = get_local_id(0);
    
    unsigned int number_of_zeros_total = n - prefix_cnt[n / 128 - 1];
    if (n/128 - 1 >= 262144 || n/128 - 1 < 0)
        printf("Radix counters_pref_gpu index1: %d \n", n/128 - 1);
    unsigned int number_of_ones_global;
    __local unsigned int number_of_ones;
    __local unsigned int part[128];


    if (id < n) {
        if (id_inside_block >= 128 || id_inside_block < 0)
            printf("Radix part(128) index1: %d \n", id_inside_block);
        if (id >= 33554432 || id < 0)
            printf("Radix as index1: %d \n", id);
        part[id_inside_block] = as[id];

        barrier(CLK_LOCAL_MEM_FENCE);

        if (id == block_id * 128) {
            number_of_ones_global = 0;
            if (block_id != 0) {
                if (block_id >= 262145 || block_id < 1)
                    printf("Radix counters_pref_gpu index2: %d \n", block_id - 1);
                number_of_ones_global = prefix_cnt[block_id - 1];
            }
            if (block_id >= 262144 || block_id < 0)
                printf("Radix counters_pref_gpu index3: %d \n", block_id);
            number_of_ones = prefix_cnt[block_id] - number_of_ones_global;
            sort_inside_block(part, &number_of_ones, current_bit);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned int number_of_zeros_loc = 128 - number_of_ones;
        unsigned int number_of_zeros_glob = block_id * 128 - number_of_ones_global;
             
        unsigned int cur_zero_pos = number_of_zeros_glob + id_inside_block;
        unsigned int cur_one_pos = number_of_zeros_total + number_of_ones_global + id_inside_block - number_of_zeros_loc;

        if (id_inside_block < number_of_zeros_loc){
            if (id_inside_block >= 128)
                printf("Radix part res1: %d \n", id_inside_block);
            if (cur_zero_pos >= 33554432 || cur_zero_pos < 0)
                printf("Radix bs1 res1: %d \n", cur_zero_pos);
            res[cur_zero_pos] = part[id_inside_block];
        }
        else{
            if (id_inside_block >= 128)
                printf("part res2: %d \n", id_inside_block);
            if (cur_one_pos >= 33554432 || cur_one_pos < 0)
                printf("Radix bs2 res2: %d \n", cur_one_pos);
            res[cur_one_pos] = part[id_inside_block];
        }
    }
}