#line 2

unsigned int binsearch_left(__global const float *as, int left, int right, float x)
{
    while(right - left > 1)
    {
        int m = (left + right) / 2;
        if (x <= as[m])
        {
            right = m;
        }
        else
        {
            left = m;
        }
    }
    return right;
}

unsigned int binsearch_right(__global const float *as, int left, int right, float x)
{
    while(right - left > 1)
    {
        int m = (left + right) / 2;
        if (x < as[m])
        {
            right = m;
        }
        else
        {
            left = m;
        }
    }
    return right;
}

__kernel void merge(__global const float *as, __global float *bs, const unsigned int n, const unsigned int block_size)
{
    unsigned int i = get_global_id(0);
    if (i >= n)
    {
        return;
    }
    float curr_elem = as[i];

    int block_offset = i % block_size;
    int block_start = i - block_offset;
    int block_end = block_start + block_size;
    int block_no = block_start / block_size;
    int if_second = block_no % 2;

    int neigh_start = block_start + (if_second ? -block_size : block_size);
    int neigh_end = block_end + (if_second ? -block_size : block_size);

    unsigned int pos;
    if (if_second)
    {
        unsigned int pos_neigh = binsearch_right(as, neigh_start - 1, neigh_end, curr_elem);
        pos = block_offset + pos_neigh;  // pos_neigh = first block start + offset
//        printf("%d to %d\nbo=%d, bs=%d, be=%d, bn=%d, if=%d, ns=%d, ne=%d, pn=%d, pos=%d\n", i, pos, block_offset, block_start, block_end, block_no,
//                                                                    if_second, neigh_start, neigh_end, pos_neigh, pos);
    }
    else
    {
        unsigned int pos_neigh = binsearch_left(as, neigh_start - 1, neigh_end, curr_elem);
        pos = block_start + block_offset + (pos_neigh - neigh_start);
//        printf("%d to %d\nbo=%d, bs=%d, be=%d, bn=%d, if=%d, ns=%d, ne=%d, pn=%d, pos=%d\n", i, pos, block_offset, block_start, block_end, block_no,
//                                                                    if_second, neigh_start, neigh_end, pos_neigh, pos);
    }
    bs[pos] = curr_elem;
}