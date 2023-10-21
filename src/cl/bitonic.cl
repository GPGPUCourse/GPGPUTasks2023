__kernel void bitonic(__global float *as, unsigned int n, unsigned int step, unsigned int size) {
    // 'n' and 'size' should be a power of 2 (we can add inf's to as to increase 'n')

    int global_i = get_global_id(0);
    global_i = global_i / (size / 2) * size + global_i % (size / 2);
    int group_number = global_i / step;
    int mult = (int)(group_number % 2 == 0) * 2 - 1;

    if (global_i < n && as[global_i] * mult > as[global_i + size / 2] * mult) {
        float tmp = as[global_i + size / 2];
        as[global_i + size / 2] = as[global_i];
        as[global_i] = tmp;
    }
}
