__kernel void prefix_sum(__global unsigned int* as, __global unsigned int* res, unsigned int n, unsigned int depth, unsigned int compress) {
    // 'n' and should be a power of 2 (we can add inf's to as to increase 'n')

    int i = get_global_id(0);
    int j = ((i + 1) >> depth) % 2;

    if (i < n) {
        res[i] += j * as[i / compress * 2];
    }
}


__kernel void compress(__global unsigned int* as, __global unsigned int* bs, unsigned int n) {
    // 'n' and should be a power of 2 (we can add inf's to as to increase 'n')

    int i = get_global_id(0);

    if (2 * i < n) {
        bs[i] = as[2 * i] + as[2 * i + 1];
    }
}