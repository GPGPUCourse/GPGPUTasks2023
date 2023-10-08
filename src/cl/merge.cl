__kernel void merge(__global const float *as,
                    __global float *bs,
                    uint sub_size,
                    int size) {
    unsigned int i = get_global_id(0);
    if (i >= size) {
        return;
    }

    unsigned int block_number = (i / sub_size);
    bool is_right = block_number & 1;
    unsigned int cur_index = i % sub_size;

    int offset = (block_number ^ 1) * sub_size;
    int left = offset - 1;
    int right = offset + sub_size;
    float value = as[i];

    if (left >= size) {
        left = offset - 1;
        right = offset;
    } else if (right > size) {
        right = size;
    }

    while (right - left > 1) {
        int mid = (left + right) >> 1;
        if (is_right && as[mid] > value || !is_right && as[mid] >= value) {
            right = mid;
        } else {
            left = mid;
        }
    }
    
    bs[i / (2 * sub_size) * (2 * sub_size) + cur_index + right - offset] = value;
}
