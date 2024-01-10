#line 2

__kernel void matrix_multiplication_basic(__global const float *a, __global const float *b, __global float *c,
                                          unsigned int M, unsigned int K, unsigned int N)
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);

    if (i >= N || j >= M)
    {
        return;
    }

    float sum = 0.0;
    for (size_t it = 0; it < K; it++)
    {
        sum += a[j * K + it] * b[it * N + i];
    }
    c[j * N + i] = sum;
}

#define TILE_SIZE 16
__kernel void matrix_multiplication_local(__global const float *a, __global const float *b, __global float *c,
                                          unsigned int M, unsigned int K, unsigned int N)
{
    size_t i = get_global_id(0);
    size_t j = get_global_id(1);
    size_t local_i = get_local_id(0);
    size_t local_j = get_local_id(1);
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (size_t tileK = 0; tileK * TILE_SIZE < K; tileK++)
    {
        if (j < M && (tileK * TILE_SIZE + local_i) < K)
        {
            tileA[local_j][local_i] = a[j * K + tileK * TILE_SIZE + local_i];
        }

        if (i < N && tileK * TILE_SIZE + local_j < K)
        {
            tileB[local_j][local_i] = b[(tileK * TILE_SIZE + local_j) * N + i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (size_t k = 0; k < TILE_SIZE; k++)
        {
            sum += tileA[local_j][k] * tileB[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (j < M && i < N)
    {
        c[j * N + i] = sum;
    }
}

#define THREAD_WORK 4
__kernel void matrix_multiplication_local_work(__global const float *a, __global const float *b, __global float *c,
                                               unsigned int M, unsigned int K, unsigned int N)
{
    size_t local_i = get_local_id(0);
    size_t local_j = get_local_id(1);
    size_t i = get_global_id(0);
    size_t j = get_group_id(1) * TILE_SIZE + local_j;

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];
    const size_t WORK_STEP = TILE_SIZE / THREAD_WORK;

    float sum[THREAD_WORK] = { 0 };

    for (size_t tileK = 0; tileK * TILE_SIZE < K; tileK++)
    {
        const size_t offset = tileK * TILE_SIZE;
        for (size_t w = 0; w * WORK_STEP < TILE_SIZE; w++)
        {
            const size_t work_offset = w * WORK_STEP;
            if (j + work_offset < M && offset + local_i < K)
            {
                tileA[local_j + work_offset][local_i] = a[(j + work_offset) * K + offset + local_i];
            }

            if (i < N && offset + local_j + work_offset < K)
            {
                tileB[local_j + work_offset][local_i] = b[(offset + local_j + work_offset) * N + i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        const size_t end = (TILE_SIZE < (K - offset)) ? TILE_SIZE : (K - offset);
        for (size_t k = 0; k < end; k++)
        {
            const float elem_b = tileB[k][local_i];
            for (size_t w = 0; w < THREAD_WORK; w++)
            {
                if (i < N && j + w * WORK_STEP < M)
                {
                    sum[w] += elem_b * tileA[local_j + w * WORK_STEP][k];
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (size_t w = 0; w < THREAD_WORK; w++)
    {
        if (i < N && j + WORK_STEP * w < M)
        {
            c[(j + WORK_STEP * w) * N + i] = sum[w];
        }
    }
}