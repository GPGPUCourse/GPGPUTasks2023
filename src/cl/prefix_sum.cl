__kernel void prefix_sum_up(__global unsigned int *as, unsigned int n, unsigned int d) {
    unsigned int gid = get_global_id(0);
    unsigned int k = (gid << (d + 1));
    as[k + (1 << (d + 1)) - 1] += as[k + (1 << d) - 1];
}

__kernel void prefix_sum_down(__global unsigned int *as, unsigned int n, unsigned int d) {
    unsigned int gid = get_global_id(0);
    unsigned int k = (gid << (d + 1));
    unsigned int tmp = as[k + (1 << d) - 1];
    as[k + (1 << d) - 1] = as[k + (1 << (d + 1)) - 1];
    as[k + (1 << (d + 1)) - 1] += tmp;
}

__kernel void set_0_as_zero(__global unsigned int *as, unsigned int n) {
    as[n - 1] = 0;
}

__kernel void shift(__global unsigned int *as, __global unsigned int *bs, unsigned int n) {
    int i = get_global_id(0);
    if (i)
        bs[i - 1] = as[i];
    else
        bs[n - 1] = as[n - 1];
}
