__kernel void merge(__global const float *as_gpu, __global float *bs_gpu, const int n,
                    const int merge_block_size) {
    const int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    int left = 0;
    int right = 0;
    bool is_block_even = i % (merge_block_size * 2) < merge_block_size;
    if (is_block_even) {
        left = i / merge_block_size * merge_block_size + merge_block_size;
    } else {
        left = i / merge_block_size * merge_block_size - merge_block_size;
    }
    if (left >= n) {
        return;
    }
    int start_left = left;
    right = min(left + merge_block_size, n);

    while (left < right) {
        int mid = left + (right - left) / 2;
        if (as_gpu[mid] > as_gpu[i] || is_block_even && as_gpu[mid] >= as_gpu[i]) {
            right = mid;
        } else {
            left = mid + 1;
        }
    }
    bs_gpu[(left - start_left) + i % merge_block_size + i / (2 * merge_block_size) * 2 * merge_block_size] = as_gpu[i];
}
