__kernel void prefix_sum(__global unsigned int *as, unsigned int n, unsigned int d) {
    unsigned int gid = get_global_id(0) + (1 << d);
    if (gid >= n)
        return;
    
    as[gid] += as[gid - (1 << (d - 1))];
}