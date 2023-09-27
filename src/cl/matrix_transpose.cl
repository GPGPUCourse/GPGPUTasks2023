#ifndef LOCAL_SIZE
    #define LOCAL_SIZE 16
#endif

__kernel void matrix_transpose(__global const float *as, __global float *as_tr, unsigned int M, unsigned int K) {
    unsigned int local_j = get_local_id(0), local_i = get_local_id(1);
    unsigned int j = get_global_id(0), i = get_global_id(1);
    unsigned int start_j = j - local_j, start_i = i - local_i;

    __local float buffer[LOCAL_SIZE][LOCAL_SIZE + 1];// fix bank conflict

    if (j < K && i < M)
        buffer[local_i][local_j] = as[i * K + j];// coalesed read

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int dest_i = start_j + local_i;
    unsigned int dest_j = start_i + local_j;

    if (dest_j < K && dest_i < M) {
        as_tr[dest_i * M + dest_j] = buffer[local_j][local_i];// coalesed write
    }
}


__kernel void matrix_transpose_bad_banks(__global const float *as, __global float *as_tr, unsigned int M,
                                         unsigned int K) {
    unsigned int local_j = get_local_id(0), local_i = get_local_id(1);
    unsigned int j = get_global_id(0), i = get_global_id(1);
    unsigned int start_j = j - local_j, start_i = i - local_i;

    __local float buffer[LOCAL_SIZE][LOCAL_SIZE];// no fix bank conflict :(
    if (j < K && i < M)
        buffer[local_i][local_j] = as[i * K + j];// coalesed read

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int dest_i = start_j + local_i;
    unsigned int dest_j = start_i + local_j;
    if (dest_j < K && dest_i < M)
        as_tr[dest_i * M + dest_j] = buffer[local_j][local_i];// coalesed write
}


__kernel void matrix_transpose_naive(__global const float *as, __global float *as_tr, unsigned int M, unsigned int K) {
    unsigned int j = get_global_id(0), i = get_global_id(1);
    if (j < K && i < M)
        as_tr[j * M + i] = as[i * K + j];
}