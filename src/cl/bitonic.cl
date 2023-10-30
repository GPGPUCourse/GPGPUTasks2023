void swap(__global float *a, __global float *b) {
    float tmp = *a;
    *a = *b;
    *b = tmp;
}

__kernel void bitonic(__global float *as, unsigned int a, unsigned int b) {
    int i = get_global_id(0);
    int t = i % a;
    int x = i - t;
    t = a - t - 1 + ((i / a) & 1) * (2 * t - a + 1);
    int k = t - b;
    k = a - k - 1 + ((i / a) & 1) * (2 * k - a + 1);
    if ((t & b) && as[i] > as[x + k])
        swap(&as[i], &as[x + k]);
}
