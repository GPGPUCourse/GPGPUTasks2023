#line 2

__kernel void bitonic(__global float *as, unsigned int n, unsigned int block_size, unsigned int gap) {
    unsigned int lower_i = get_global_id(0);
    unsigned int higher_i = lower_i ^ gap;
    if (higher_i > lower_i)
    {
        float lower = as[lower_i];
        float higher = as[higher_i];
        if (lower_i & block_size)
        {
            if (lower < higher)
            {
                as[lower_i] = higher;
                as[higher_i] = lower;
            }
        }
        else
        {
            if (lower > higher)
            {
                as[lower_i] = higher;
                as[higher_i] = lower;
            }
        }
    }
}
