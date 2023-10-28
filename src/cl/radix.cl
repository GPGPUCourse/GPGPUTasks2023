#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void save(__global unsigned int *as,
                   __global unsigned int *t,
                   unsigned int d,
                   unsigned int n)
{
    unsigned int i = get_global_id(0);
    if (i < n)
    {
        for (int x = 0; x < MAX_DIGIT; x++)
        {
            t[i + n * x] = 0;
        }
        t[i + n * (as[i] >> (4 * d) & 15)] = 1;
    }
}

__kernel void sums1(__global unsigned int *t,
                    __global unsigned int *t2,
                    const unsigned int len,
                    const unsigned int n)
{
    unsigned int pos = get_global_id(0) * len + len - 1;
    if (pos >= n) return;

    for (unsigned int x = 0; x < MAX_DIGIT; x++)
    {
        if (len == 1) t2[pos + x * n] = t[pos + x * n];
        else t2[pos + x * n] += t2[pos + x * n - len / 2];
    }
}

__kernel void radix(__global unsigned int *as,
                    __global unsigned int *bs,
                    __global unsigned int *t2,
                    const unsigned int d,
                    const unsigned int n)
{
    unsigned int i = get_global_id(0);
    if (i < n)
    {
        unsigned int x = (as[i] >> (4 * d) & 15);
        unsigned int calc = 0;
        for (unsigned int y = 0; y < x; y++) calc += t2[y * n + n - 1];
        int j = i;
        while (j >= 0)
        {
            calc += t2[x * n + j];
            j = (j & (j + 1)) - 1;
        }
        bs[calc - 1] = as[i];
    }
}
