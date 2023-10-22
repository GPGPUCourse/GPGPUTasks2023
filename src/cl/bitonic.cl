#define WORKGROUP_SIZE 32

__kernel void bitonic(__global float *as, unsigned int n_log, unsigned int k_log, unsigned int step) {
    unsigned int i = get_global_id(0);
    if (i >= (1 << n_log >> 1)) return;

    unsigned int base = 2 * i;
    unsigned int reverse_order = ((base >> k_log) & 1);
    unsigned int block_size = (1 << step);
    unsigned int block_begin = (base >> step << step);
    unsigned int first_pos = block_begin + ((base - block_begin) >> 1);
    unsigned int second_pos = first_pos + (block_size >> 1);

    if ((as[first_pos] > as[second_pos]) ^ reverse_order) {
        float tmp = as[second_pos];
        as[second_pos] = as[first_pos];
        as[first_pos] = tmp;
    }
}
