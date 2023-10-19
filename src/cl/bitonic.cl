__kernel void bitonic(__global float *as, const unsigned int n, const unsigned int cur_block_size,
                      const unsigned int j) {
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    bool is_reversed = i % (cur_block_size << 1) >= cur_block_size;
    unsigned int cur_i = (i / j) * (j << 1) + (i % j);
    if (cur_i + j < n && (is_reversed && as[cur_i] < as[cur_i + j] || !is_reversed && as[cur_i] > as[cur_i + j])) {
        const float tmp = as[cur_i];
        as[cur_i] = as[cur_i + j];
        as[cur_i + j] = tmp;
    }
}
