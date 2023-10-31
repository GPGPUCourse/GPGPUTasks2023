__kernel void update_blocks(__global unsigned int *bs,
                            unsigned int block_size,
                            unsigned int n)
{
    int i = get_global_id(0);
    if(i * block_size + block_size / 2 < n)
        bs[i * block_size] += bs[i * block_size + block_size / 2];
}

__kernel void prefix_sum(__global unsigned int *as,
                        __global unsigned int *bs,
                        unsigned int block_size,
                        unsigned int n)
{
    int i = get_global_id(0);
    if(i < n && ((i + 1)&block_size) != 0) {
        as[i] += bs[((i + 1) / block_size - 1) * block_size];
    }
}