bool cmp(float fst, float snd, bool is_fst_half) {
    if (is_fst_half)
        return fst < snd;
    else
        return fst <= snd;
}

int binary_search(const __global float *a, int len, float value, bool is_fst_half) {
    int left = -1;
    int right = len;
    while (right - left > 1) {
        int mid = (left + right) / 2;
        if (cmp(a[mid], value, is_fst_half))
            left = mid;
        else
            right = mid;
    }

    return left + 1;
}

__kernel void merge(const __global float *a, __global float *c, unsigned int half_block_size) {
    unsigned int id = get_global_id(0);
    unsigned int block = id / (2 * half_block_size);
    unsigned int c_id = 0;
    if ((id / half_block_size) % 2 == 0)
        c_id = id + binary_search(a + (block * 2 + 1) * half_block_size, half_block_size, a[id], false);
    else
        c_id = id - half_block_size + binary_search(a + block * 2 * half_block_size, half_block_size, a[id], true);
    c[c_id] = a[id];
}