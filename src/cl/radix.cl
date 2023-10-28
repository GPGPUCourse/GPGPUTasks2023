#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sums(__global unsigned int *as,
                   __global unsigned int *t,
                   const unsigned int x,
                   const unsigned int d,
                   const unsigned int len,
                   const unsigned int n)
{
    unsigned int pos = get_global_id(0) * len + len - 1;
    if (pos >= n) return;

    if (len == 1) t[pos] = (as[pos] >> (LOG_MAX_DIGIT * d) & (MAX_DIGIT - 1)) == x;
    else t[pos] += t[pos - len / 2];
}

__kernel void radix(__global unsigned int *as,
                    __global unsigned int *bs,
                    __global unsigned int *t,
                    const unsigned int pnt,
                    const unsigned int n,
                    const unsigned int w)
{
    unsigned int i = get_global_id(0);
    if (i >= w) return;

    unsigned int j = i;
    int pos = -1;
    for (int len = n / 2; len >= 1; len >>= 1)
    {
        if (t[pos + len] <= i)
        {
            i -= t[pos + len];
            pos += len;
        }
    }
    bs[pnt + j] = as[pos + 1];
}
