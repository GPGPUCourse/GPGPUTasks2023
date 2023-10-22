__kernel void pref_sum(__global unsigned int *src, __global unsigned int *dest, int n, int shift) {
    int id = get_global_id(0);
    if (id >= n) return;

    dest[id] = src[id] + (id >= shift ? src[id - shift] : 0);
}
