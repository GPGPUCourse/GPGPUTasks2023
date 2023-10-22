__kernel void bitonic(__global float *as, unsigned int n, unsigned int bit_block, unsigned int splitted_block) {
    int id = get_global_id(0);
    if (id >= n / 2)
        return;

    bool is_reversed = (id / bit_block) % 2;
    int splitted_block_id = id / splitted_block;
    int splitted_block_begin = splitted_block_id * splitted_block * 2;
    int this_index = splitted_block_begin + id % splitted_block;
    int other_index = this_index + splitted_block;

    float Min = min(as[this_index], as[other_index]);
    float Max = max(as[this_index], as[other_index]);

    if (!is_reversed) {
        as[this_index] = Min;
        as[other_index] = Max;
    } else {
        as[this_index] = Max;
        as[other_index] = Min;
    }
}
