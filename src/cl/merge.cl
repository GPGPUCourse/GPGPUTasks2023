int binary_search(const __global float *a, unsigned int len, float value) {
    int left = -1;
    int right = len;
    while (right - left > 1) {
        int mid = (left + right) / 2;
        if (a[mid] < value)
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

    if (id / half_block_size % 2 == 0)
        c_id = id + binary_search(a + (block * 2 + 1) * half_block_size, half_block_size, a[id]);
    else
        c_id = id - half_block_size + binary_search(a + block * 2 * half_block_size, half_block_size, a[id]);    

    c[c_id] = a[id];
}