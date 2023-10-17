__kernel void prefix_sum(__global float *as, __global float *bs, const unsigned int i , const unsigned int n) {
    int global_i = get_global_id(0);
    if (global_i < n) {
        bs[global_i] = as[global_i];
        int j = global_i - i;
        if (j >= 0) {
            bs[global_i] += as[global_i];
        }
    }
}