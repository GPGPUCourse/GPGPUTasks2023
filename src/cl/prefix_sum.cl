__kernel void prefix_sum(__global unsigned int* as, __global unsigned int* res, unsigned int n, unsigned int depth, unsigned int compress) {
    // 'n' and should be a power of 2 (we can add inf's to as to increase 'n')

    int i = get_global_id(0);
    int j = ((i + 1) >> depth) % 2;

    if (i < n) {
        res[i] += j * as[(i / compress) * compress];
    }
}

__kernel void compress(__global unsigned int* as, unsigned int n, unsigned int compress) {
    // 'n' and should be a power of 2 (we can add inf's to as to increase 'n')

    int i = get_global_id(0);
    int j = i + compress / 2;

    if (i % compress == 0 && j < n) {
        as[i] += as[j];
    }
}