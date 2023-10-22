__kernel void bitonic(__global float *as, unsigned int block, unsigned int offset) {
    int id = get_global_id(0);
    int block_id = id / offset;
    int i = id % offset + block_id * (offset << 1);
    int right = (i / block) & 1;
    float temp = 0;
    if (!right && as[i + offset] < as[i] || right && as[i] < as[i + offset]) {
        temp = as[i + offset];
        as[i + offset] = as[i];
        as[i] = temp;
    }
}
