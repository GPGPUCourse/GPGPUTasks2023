void swap(__global float *a, __global float *b) {
    float tmp = *a;
    *a = *b;
    *b = tmp;
}

__kernel void bitonic(__global float *as, unsigned int n) {
    int i = get_global_id(0);
    for(int a = 2; a < (n << 1); a <<= 1) {
        //printf("%d\n", a);
        for (int b = (a >> 1); b >= 1; b >>= 1) {
            int t = i % a;
            int x = i - t;
            t = a - t - 1 + ((i / a) & 1) * (2 * t - a + 1);
            int k = t - b;
            k = a - k - 1 + ((i / a) & 1) * (2 * k - a + 1);
            if ((t & b) && i < n && x + k < n && as[i] > as[x + k])
                swap(&as[i], &as[x + k]);
            barrier(CLK_LOCAL_MEM_FENCE);

            /*if(i == 0) {
                printf("%d, %d: %f %f %f %f %f %f %f %f %f %f\n",
                    a, b,
                    as[0], as[1], as[2], as[3], as[4],
                    as[5], as[6], as[7], as[8], as[9]);
            }*/
        }
    }
}
