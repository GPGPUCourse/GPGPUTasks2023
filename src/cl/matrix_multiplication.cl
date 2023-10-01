__kernel void matrix_multiplication_trivial(__global const float* a , __global const float* b, __global float* res,
        unsigned int M, unsigned int K, unsigned int N)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);
    
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        sum += a[i * K + k] * b[k * N + j];
    }
    if (i < M && j < N) res[i * N + j] = sum;
}

// Keep value synced with same define inside main_matrix_transpose.cpp
#define WGS 16u

__kernel void matrix_multiplication_local_mem(__global const float* a , __global const float* b, __global float* res,
        unsigned int M, unsigned int K, unsigned int N)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    const int local_i = get_local_id(0);
    const int local_j = get_local_id(1);

    const int gid_i = get_group_id(0);
    const int gid_j = get_group_id(1);
    
    // second dimension extended to prevent bank conflicts when we read whole column in one cycle
    __local float tileA[WGS + 1][WGS];
    __local float tileB[WGS + 1][WGS];
    
    float sum = 0.f;
    for (int tileK = 0; tileK < K; tileK += WGS) {

        if (i < M && tileK + local_j < K) tileA[local_i][local_j] = a[i * K + (tileK + local_j)];
        if (tileK + local_i < K && j < N) tileB[local_i][local_j] = b[(tileK + local_i) * N + j];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < WGS; ++k) {
            sum += tileA[local_i][k] * tileB[k][local_j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < M && j < N) res[i * N + j] = sum;
}
