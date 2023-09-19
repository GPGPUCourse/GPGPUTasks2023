typedef unsigned int uint_;

__kernel void sum1(__global const uint_* src, __global uint_* res, const uint_ size) {
    const uint_ id = get_global_id(0);

    if (id >= size) {
        return;
    }

    atomic_add(res, src[id]);
}