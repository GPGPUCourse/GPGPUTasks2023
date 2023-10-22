#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sums1(__global unsigned int *as,
                    __global unsigned int *bs,
                    const unsigned int len,
                    const unsigned int n)
{
    unsigned int pos = get_global_id(0) * len + len - 1;
    if (pos >= n) return;

    if (len == 1) bs[pos] = as[pos];
    else bs[pos] += bs[pos - len / 2];
}

__kernel void sums2(__global unsigned int *as,
                    __global unsigned int *bs,
                    const unsigned int n)
{
    int pos = get_global_id(0);
    if (pos >= n) return;

    int i = pos;
    as[i] = 0;
    while (pos >= 0)
    {
        as[i] += bs[pos];
        pos &= (pos + 1);
        pos--;
    }
}
