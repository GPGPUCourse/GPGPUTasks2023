// vim: syntax=c


__kernel void merge_naive(
    __global const float * a,
    __global float * b,
    unsigned int k
) {
    const unsigned int id = get_global_id(0);
    const unsigned int base_idx = 2 * k * id;

    __global const float * const a1 = a + base_idx;
    __global const float * const a2 = a1 + k;

    __global float * ptr = b + base_idx;
    for (unsigned int i = 0, j = 0; i < k || j < k; ) {
        if (i >= k || j < k && a2[j] < a1[i]) {
            *ptr = a2[j];
            ++j;
        } else {
            *ptr = a1[i];
            ++i;
        }

        ++ptr;
    }
}
