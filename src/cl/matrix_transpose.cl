#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIZE 16

__kernel void matrix_transpose(__global const float * as,
                               __global float * as_t,
                               const unsigned int m,
                               const unsigned int k)
{
    int li = get_local_id(1);
    int lj = get_local_id(0);
    int gi = get_group_id(1);
    int gj = get_group_id(0);

    __local float tmp[GROUP_SIZE][GROUP_SIZE];
    int i = gi * GROUP_SIZE + li;
    int j = gj * GROUP_SIZE + lj;
    if (i < m && j < k)
    {
        tmp[lj][li] = as[i * k + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    i = gj * GROUP_SIZE + li;
    j = gi * GROUP_SIZE + lj;
    if (i < k && j < m)
    {
        as_t[i * m + j] = tmp[li][lj];
    }
}

__kernel void matrix_transpose_not_coalesced(__global const float * as,
                                             __global float * as_t,
                                             const unsigned int m,
                                             const unsigned int k)
{
    int li = get_local_id(0);
    int lj = get_local_id(1);
    int gi = get_group_id(0);
    int gj = get_group_id(1);

    __local float tmp[GROUP_SIZE][GROUP_SIZE];
    int i = gi * GROUP_SIZE + li;
    int j = gj * GROUP_SIZE + lj;
    if (i < m && j < k)
    {
        tmp[lj][li] = as[i * k + j];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    i = gj * GROUP_SIZE + li;
    j = gi * GROUP_SIZE + lj;
    if (i < k && j < m)
    {
        as_t[i * m + j] = tmp[li][lj];
    }
}

__kernel void matrix_transpose_naive(__global const float * as,
                                     __global float * as_t,
                                     const unsigned int m,
                                     const unsigned int k)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    if (i < m && j < k)
    {
        as_t[j * m + i] = as[i * k + j];
    }
}