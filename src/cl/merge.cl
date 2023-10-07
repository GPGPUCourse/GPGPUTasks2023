__kernel void merge(const __global float *as_gpu, __global float *bs_gpu, const unsigned int n,
                    const unsigned int merge_block_size) {
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    int even = i / (2 * merge_block_size) * (2 * merge_block_size);
    int odd = even + merge_block_size;
    bool is_block_even = i < odd;
    int left = is_block_even ? odd : even;
    int start_left = left;
    int right = left + merge_block_size < n ? left + merge_block_size : n;
    if (left >= n) {
        return;
    }
    

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (as_gpu[mid] > as_gpu[i] || is_block_even && as_gpu[mid] >= as_gpu[i]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    int final_i = (left - start_left) + i % merge_block_size + i / (2 * merge_block_size) * 2 * merge_block_size;
    if (final_i < n) {
        bs_gpu[final_i] = as_gpu[i];
    }
}
