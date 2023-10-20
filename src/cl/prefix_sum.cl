// vim: filetype=c


__kernel void prefix_sum(__global unsigned int * as, unsigned int mask) {
    const unsigned int i = get_global_id(0);

    if (mask & i) {
        as[i] += as[i / mask * mask - 1];
    }
}
