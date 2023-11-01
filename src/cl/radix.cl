__kernel void prefix_sum(__global const unsigned int *as, __global unsigned int *bs, unsigned int n, unsigned int block_size) {
    int id = get_global_id(0);
    if (id < n && ((id + 1) & block_size))
        bs[id] += as[((id + 1) / block_size) - 1];
}

__kernel void reduce(__global const unsigned int *as, __global unsigned int *res, unsigned int n) {
    int id = get_global_id(0);
    if (id < n)
        res[id] = as[id * 2] + as[id * 2 + 1];
}

__kernel void cnt_calc(__global const unsigned int *as, __global unsigned int *counters, unsigned int current_bit, unsigned int n) {
    int id = get_global_id(0);
    if (id < n) {
        counters[id] = 0;
        int start = id * 128;
        for (int i = start; i < start + 128; i++)
            if (as[i] & (1 << current_bit))
                counters[id]++;
    }
}

void sort_inside_block(__local unsigned int *part, __local unsigned int *number_of_zeros, unsigned int current_bit) {
    unsigned int part_cnt = 0;
    unsigned int part_copy[128];
    for (unsigned int i = 0; i < 128; ++i) {
        part_copy[i] = part[i];
        if ((part_copy[i] >> current_bit) & 1)
            part_cnt++;
    }

    unsigned int current_one = 128 - part_cnt;
    unsigned int current_zero = 0;

    for (unsigned int i = 0; i < 128; ++i)
        if (part_copy[i] & (1 << current_bit))
            part[current_one++] = part_copy[i];
        else
            part[current_zero++] = part_copy[i];
    *number_of_zeros = part_cnt;
}

__kernel void radix(__global const unsigned int *prefix_cnt, __global unsigned int *as, __global unsigned int *res,
                        unsigned int current_bit, unsigned int n) {
    const unsigned int id = get_global_id(0);
    const unsigned int block_id = get_group_id(0);
    const unsigned int id_inside_block = get_local_id(0);
    
    unsigned int number_of_ones_global;
    __local unsigned int number_of_zeros;
    __local unsigned int part[128];
   
   
   if (id < n) {
        part[id_inside_block] = as[id];

        barrier(CLK_LOCAL_MEM_FENCE);

        if (id == block_id * 128) {
            number_of_ones_global = 0;
            if (block_id != 0)
                number_of_ones_global = prefix_cnt[block_id - 1];
            sort_inside_block(part, &number_of_zeros, current_bit);
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        unsigned int number_of_zeros_loc = 128 - number_of_zeros;
        unsigned int number_of_zeros_glob = block_id * 128 - number_of_ones_global;
        unsigned int number_of_zeros_total = n - prefix_cnt[n / 128 - 1];
             
        unsigned int cur_zero_pos = number_of_zeros_glob + id_inside_block;
        unsigned int cur_one_pos = number_of_zeros_total + number_of_ones_global + id_inside_block - number_of_zeros_loc;

        barrier(CLK_GLOBAL_MEM_FENCE);

        if (id_inside_block < number_of_zeros_loc)
            res[cur_zero_pos] = part[id_inside_block];
        else
            res[cur_one_pos] = part[id_inside_block];
    }
}