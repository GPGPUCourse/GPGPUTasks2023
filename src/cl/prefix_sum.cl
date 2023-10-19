

__kernel void reduce(__global const unsigned int *from, __global unsigned int *to) {
    int i = get_global_id(0);
    to[i] = from[i * 2] + from[i * 2 + 1];
}

__kernel void sum_if_need(__global const unsigned int *pre_sums, __global unsigned int *scan, int sum_len) {
    int i = get_global_id(0);
    i++;
    if (i & sum_len) {
        scan[i - 1] += pre_sums[i / sum_len - 1];
    }
}