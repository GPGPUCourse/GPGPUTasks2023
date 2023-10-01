// Keep value synced with same define inside main_matrix_transpose.cpp
#define WGS 16

__kernel void matrix_transpose(__global const float* data, __global float* res, unsigned int M, unsigned int K)
{

    // second dimension extended to prevent bank conflicts when we read whole column in one cycle
    __local float buff[WGS][WGS + 1];

    const int i = get_global_id(0);
    const int j = get_global_id(1);

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);

    const int gid_i = get_group_id(0);
    const int gid_j = get_group_id(1);


    // coalesced read, no bank conflicts
    if (i < K && j < M) buff[local_i][local_j] = data[j * K + i];

    barrier(CLK_LOCAL_MEM_FENCE);

    // non-coalesced write
    // res[i * M + j] = buff[local_i][local_j];

    // coalesced write
    unsigned int new_i = gid_i * WGS + local_j;
    unsigned int new_j = gid_j * WGS + local_i;
    if (new_i < K && new_j < M) res[new_i * M + new_j] = buff[local_j][local_i];
}
