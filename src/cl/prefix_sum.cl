__kernel void prefix_sum(__global unsigned int *as, __global unsigned int *result, const unsigned int n,
                         const unsigned int cur_block_size) {
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    if ((i + 1) % cur_block_size == 0 && (cur_block_size != 1)) {
        as[i] += as[i - cur_block_size / 2];
    }
    if (((i + 1) & cur_block_size) != 0) {
        unsigned int offset = i;
        unsigned int mask = -1;
        while (offset + cur_block_size > i + 1) {
            offset &= mask;
            mask <<= 1;
        }
        result[i] += as[cur_block_size + offset - 1];
        // if (i == 0) {
        //     printf("%d %d %d\n", cur_block_size + offset - 1, cur_block_size, mask);
        // }
    }
}

// __kernel void prefix_sum_reduce(__global float *as, const unsigned int n, const unsigned int cur_block_size) {
//     const unsigned int i = get_global_id(0);
//     if (((i + 1) * cur_block_size) - 1 >= n) {
//         return;
//     }
//     as[((i + 1) * cur_block_size) - 1] += as[i * cur_block_size - 1];
// }