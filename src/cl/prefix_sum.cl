// vim: filetype=c


__kernel void prefix_sum(__global unsigned int * as, unsigned int m) {
    const unsigned int id = get_global_id(0);

    const unsigned int i = (id / m) * 2 * m + (id % m) + m;
    as[i] += as[i / m * m - 1];
}
