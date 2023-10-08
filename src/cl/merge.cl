#ifdef __Clocal_yON_IDE__
    #include <local_ybgpu/opencl/cl/clocal_yon_defines.cl>
#endif

#line 6

__kernel void merge(__global float *as, __global float *tmp, unsigned int size, unsigned int n) {
    unsigned int i = get_global_id(0);
    if (i >= n)
        return;

    const float a = as[i];
    unsigned int offset = i / (2 * size) * 2 * size;
    unsigned int in_left = (i - offset < size);
    unsigned int in_group_offset = i - offset - size + in_left * size;

    int m = 0;
    int left = -1;
    int right = size;
    const unsigned int true_index = offset + in_left * size;

    while (right - left > 1) {
        m = (right + left) / 2;
        if (in_left ? as[m + true_index] > a : as[m + true_index] >= a)
            right = m;
        else
            left = m;
    }

    tmp[offset + in_group_offset + left + 1] = a;
}
