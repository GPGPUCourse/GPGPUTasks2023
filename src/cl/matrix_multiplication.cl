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

// Keep values synced with same define inside main_matrix_transpose.cpp
#define TS 32u
#define WS 8u

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
    __local float tileA[TS + 1][TS];
    __local float tileB[TS + 1][TS];
    
    float sum = 0.f;
    unsigned int new_i, new_j;
    for (int tileK = 0; tileK < K; tileK += TS) {

        new_i = gid_i * TS + local_j;
        new_j = tileK + local_i;
        if (new_i < M && new_j < K) tileA[local_j][local_i] = a[new_i * K + new_j];
        new_i = tileK + local_i;
        new_j = j;
        if (new_i < K && new_j < N) tileB[local_i][local_j] = b[new_i * N + new_j];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TS; ++k) {
            sum += tileA[local_i][k] * tileB[k][local_j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (i < M && j < N) res[i * N + j] = sum;
}

__kernel void matrix_multiplication_heavy_threads(__global const float* a , __global const float* b, __global float* res,
        unsigned int M, unsigned int K, unsigned int N)
{
    const int i = get_global_id(0);
    const int j = get_global_id(1);

    const int local_i = get_local_id(0);
    const int work_j = get_local_id(1);

    const int gid_i = get_group_id(0);
    const int gid_j = get_group_id(1);
    
    // second dimension extended to prevent bank conflicts when we read whole column in one cycle
    __local float tileA[TS + 1][TS];
    __local float tileB[TS + 1][TS];
    
    float sum[WS];
    for (int w = 0; w < WS; ++w) {
        sum[w] = 0;
    }

    unsigned int new_i, new_j;
    for (int tileK = 0; tileK < K; tileK += TS) {

        for (int w = 0; w < WS; ++w) {
            const int local_j = work_j * WS + w;
            new_i = gid_i * TS + local_j;
            new_j = tileK + local_i;
            if (new_i < M && new_j < K) tileA[local_j][local_i] = a[new_i * K + new_j];
            new_i = tileK + local_i;
            new_j = j * WS + w;
            if (new_i < K && new_j < N) tileB[local_i][local_j] = b[new_i * N + new_j];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TS; ++k) {
            float tmp = tileA[local_i][k];
            for (int w = 0; w < WS; ++w) {
                sum[w] += tmp * tileB[k][work_j * WS + w];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WS; ++w) {
        if (i < M && j * WS + w < N) res[i * N + j * WS + w] = sum[w];
    }
    
}
