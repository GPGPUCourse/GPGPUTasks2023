__kernel void bitonic(__global float *as, const unsigned int chunk, const unsigned int step_i, const unsigned int n) {
    const int global_i = get_global_id(0);
    if (global_i < n) {
        int i = 2 * global_i - global_i % step_i;
        int j = i + step_i;
        int direction = global_i / chunk;
        if (direction % 2 == 1) {
            int tmp = i;
            i = j;
            j = tmp;
        }
        if (as[i] > as[j]) {
            float tmp = as[i];
            as[i] = as[j];
            as[j] = tmp;
        }
    }
}
