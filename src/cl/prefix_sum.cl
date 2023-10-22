__kernel void reduce_a(__global unsigned int *as, __global unsigned int *bs) {
    int id = get_global_id(0);
    as[id] = bs[id * 2] + bs[id * 2 + 1];
}

__kernel void prefix_sum(__global const unsigned int *as, __global unsigned int *res, unsigned int block_size) {
    int id = get_global_id(0);
    if ((id + 1) & block_size)
        res[id] += as[(id + 1) / block_size - 1];
}