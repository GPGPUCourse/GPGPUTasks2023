#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6


__kernel void merge(const __global float *a, __global float *res, unsigned int curr_size)
{
    int i = get_global_id(0);
    int first_block_begin = (i / (2 * curr_size)) * (2 * curr_size);
    int second_block_begin = first_block_begin + curr_size;

    bool in_second = i >= second_block_begin;
    int start = first_block_begin + (!in_second) * curr_size;
    int l_value = start;
    int r_value = l_value + curr_size;

    l_value--;

    while (l_value + 1 < r_value) {
        int m_value = (l_value + r_value) / 2;
        if ((in_second && a[m_value] <= a[i]) || (!in_second && a[m_value] < a[i]))
            l_value = m_value;
        else
            r_value = m_value;
    }
    l_value -= start - 1;
    res[i + l_value - in_second * curr_size] = a[i];
}
