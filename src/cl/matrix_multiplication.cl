__kernel void naive_matrix_multiplication(__global float* a, __global float* b, __global float* c, unsigned int m, unsigned int k, unsigned int n)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    float sum = 0.0f;
    for (int t = 0; t < k; t++)
        sum += a[j * k + t] * b[t * n + i];
    c[j * n + i] = sum;
}

#define TILE_SIZE 16

__kernel void block_matrix_multiplication(__global float* a, __global float* b, __global float* c, unsigned int m, unsigned int k, unsigned int n)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tile_k = 0; tile_k < k; tile_k += TILE_SIZE) {
        tile_a[local_j][local_i] = a[j * k + tile_k + local_i];
        tile_b[local_j][local_i] = b[(local_j + tile_k) * n + i];

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < TILE_SIZE; t++)
            sum += tile_a[local_j][t] * tile_b[t][local_i];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[j * n + i] = sum;
}

#define THREAD_WORK 16

__kernel void threadwork_matrix_multiplication(__global float* a, __global float* b, __global float* c, unsigned int m, unsigned int k, unsigned int n)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tile_a[TILE_SIZE][THREAD_WORK];
    __local float tile_b[TILE_SIZE][THREAD_WORK];

    float sum[THREAD_WORK];
    for (int w = 0; w < THREAD_WORK; w++)
        sum[w] = 0;
    for (int tile_k = 0; tile_k < k; tile_k += TILE_SIZE) {
        for (int w = 0; w < THREAD_WORK; w++) {
            tile_a[local_j][w] = a[j * k + tile_k + w];
            tile_b[local_j][w] = b[(local_j + tile_k) * n + i * THREAD_WORK + w];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < TILE_SIZE; t++) {
            for (int w = 0; w < THREAD_WORK; w++)
                sum[w] += tile_a[local_j][t] * tile_b[t][w];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < THREAD_WORK; w++)
        c[j * n + i * THREAD_WORK + w] = sum[w];
}