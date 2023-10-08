__kernel void merge(
  __global const float *as        ,
  __global       float *bs        ,
  uint                  block_size,
  int                   size       ) {
    unsigned int index = get_global_id(0);
    if (index >= size) {
        return;
    }

    unsigned int block_number = (index / block_size);
    unsigned int offset = (block_number ^ 1) * block_size;
    int left = offset - 1;
    int right = offset + block_size;

    if (left >= size) {
        left = offset - 1;
        right = offset;
    } else if (right > size) {
        right = size;
    }

    while (right - left > 1) {
        int mid = (left + right) / 2;
        if (block_number % 2 == 1 && as[mid] > as[index]) {
          right = mid;
        } else if (block_number % 2 == 0 && as[mid] >= as[index]) {
          right = mid;
        } else {
          left = mid;
        }
    }

    unsigned int shift = index - index % (2 * block_size) + index % block_size;
    bs[shift + right - offset] = as[index];
}
