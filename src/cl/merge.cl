#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6


__kernel void merge(const __global unsigned int *a, __global unsigned int *res, unsigned int curr_size, unsigned int offset)
{
    int i = get_global_id(0);
    int local_id = get_local_id(0);
    int first_block_begin = (i / (2 * curr_size)) * (2 * curr_size);
    int second_block_begin = first_block_begin + curr_size;

    unsigned int elem = (a[i] >> offset) & 15;
    __local unsigned block[128];
    block[local_id] = elem;
    barrier(CLK_LOCAL_MEM_FENCE);

    bool in_second = i >= second_block_begin;
    int start = first_block_begin % 128 + (!in_second) * curr_size;
    int l_value = start;
    int r_value = l_value + curr_size;

    l_value--;

    while (l_value + 1 < r_value) {
        int m_value = (l_value + r_value) / 2;
        if ((in_second && block[m_value] <= elem) || (!in_second && block[m_value] < elem))
            l_value = m_value;
        else
            r_value = m_value;
    }
    l_value -= start - 1;
    res[i + l_value - in_second * curr_size] = a[i];
}
