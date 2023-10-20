__kernel void reduce(__global const unsigned int *bs, __global unsigned int *as, const unsigned int n) {
    int i = get_global_id(0);
    if (i < n) {
        as[i] = bs[i * 2] + bs[i * 2 + 1];
    }
}

__kernel void prefix_sum(__global const unsigned int *as, __global unsigned int *res, unsigned int i, const unsigned int n) {
    int global_i = get_global_id(0);
    if (global_i < n && ++global_i & i) {
        res[global_i - 1] += as[global_i / i - 1];
    }
}
