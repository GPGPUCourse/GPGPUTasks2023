#ifdef MT_WORK_GROUP_SIZE

__kernel void matrix_transpose(__global unsigned *a, __global unsigned *at, unsigned int width, unsigned int height)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    int local_i = get_local_id(0);
    int local_j = get_local_id(1);
    __local float tile[MT_WORK_GROUP_SIZE][MT_WORK_GROUP_SIZE];
    if (i < width && j < height)
        tile[local_j][local_i] = a[j * width + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    if (i < width && j < height)
        at[i * height + j] = tile[local_j][local_i];
}

#endif