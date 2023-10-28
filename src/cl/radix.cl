#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sums1(__global unsigned int *as,
                    __global unsigned int *t,
                    const unsigned int d,
                    const unsigned int len,
                    const unsigned int n)
{
    unsigned int pos = get_global_id(0) * len + len - 1;
    if (pos >= n) return;

    for (unsigned int x = 0; x < MAX_DIGIT; x++)
    {
        if (len == 1) t[pos + x * n] = (as[pos] >> (LOG_MAX_DIGIT * d) & (MAX_DIGIT - 1)) == x;
        else t[pos + x * n] += t[pos + x * n - len / 2];
    }
}

__kernel void radix(__global unsigned int *as,
                    __global unsigned int *bs,
                    __global unsigned int *t,
                    const unsigned int d,
                    const unsigned int n)
{
    unsigned int i = get_global_id(0);
    if (i < n)
    {
        unsigned int x = as[i] >> (LOG_MAX_DIGIT * d) & (MAX_DIGIT - 1);
        unsigned int calc = 0;
        for (unsigned int y = 0; y < x; y++) calc += t[y * n + n - 1];
        int j = i;
        while (j >= 0)
        {
            calc += t[x * n + j];
            j = (j & (j + 1)) - 1;
        }
        bs[calc - 1] = as[i];
    }
}
