__kernel void matrix_multiplication_naive(__global const float *as, __global const float *bs, __global float *cs,
                                          unsigned int M, unsigned int K, unsigned int N) {
    int i = get_global_id(1), j = get_global_id(0);
    if (i >= M || j >= N)
        return;
    float sum = 0;
    for (int k = 0; k < K; k++)
        sum += as[i * K + k] * bs[k * N + j];
    cs[i * N + j] = sum;
}

#ifndef TILE_SIZE
    #define TILE_SIZE 16
#endif

__kernel void matrix_multiplication_local_mem(__global const float *as, __global const float *bs, __global float *cs,
                                              unsigned int M, unsigned int K, unsigned int N) {
    int i = get_global_id(1), j = get_global_id(0);
    int local_i = get_local_id(1), local_j = get_local_id(0);

    __local float tile_a[TILE_SIZE][TILE_SIZE], tile_b[TILE_SIZE][TILE_SIZE];
    float sum = 0;
    for (int tile_idx = 0; tile_idx < K; tile_idx += TILE_SIZE) {
        if (tile_idx + local_j < K && i < M)
            tile_a[local_i][local_j] = as[i * K + (tile_idx + local_j)];// coalesed read as
        if (tile_idx + local_i < K && j < N)
            tile_b[local_i][local_j] = bs[(tile_idx + local_i) * N + j];// coalesed read bs
        barrier(CLK_LOCAL_MEM_FENCE);

        if (i < M && j < N) {
            for (int k = 0; k < min(TILE_SIZE, (int) (K - tile_idx)); k++) {
                sum += tile_a[local_i][k] * tile_b[k][local_j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (i < M && j < N) {
        cs[i * N + j] = sum;
    }
}


#ifndef WORK_PER_ITEM
    #define WORK_PER_ITEM 4
#endif

__kernel void matrix_multiplication_local_mem_more_work(__global const float *as, __global const float *bs,
                                                        __global float *cs, unsigned int M, unsigned int K,
                                                        unsigned int N) {
    // assume TILE_SIZE % WORK_PER_ITEM == 0
    const int ROW_TILE_SIZE = TILE_SIZE / WORK_PER_ITEM;
    int local_i = get_local_id(1), local_j = get_local_id(0);
    int i = get_group_id(1) * TILE_SIZE + local_i, j = get_global_id(0);

    __local float tile_a[TILE_SIZE][TILE_SIZE], tile_b[TILE_SIZE][TILE_SIZE];

    float sums[WORK_PER_ITEM];
    for (int i = 0; i < WORK_PER_ITEM; i++)
        sums[i] = 0.f;

    for (int tile_idx = 0; tile_idx < K; tile_idx += TILE_SIZE) {
        for (int w = 0; w < TILE_SIZE; w += ROW_TILE_SIZE) {
            if (tile_idx + local_j < K && (i + w) < M)
                tile_a[local_i + w][local_j] = as[(i + w) * K + (tile_idx + local_j)];// coalesed read as
            if ((tile_idx + local_i + w) < K && j < N)
                tile_b[local_i + w][local_j] = bs[(tile_idx + local_i + w) * N + j];// coalesed read bs
        }
        barrier(CLK_LOCAL_MEM_FENCE);


        for (int k = 0; k < min(TILE_SIZE, (int) (K - tile_idx)); k++) {
            float tile_b_value = tile_b[k][local_j];
            for (int w = 0; w < WORK_PER_ITEM; w++) {
                if (i + w * ROW_TILE_SIZE < M && j < N) {
                    sums[w] += tile_a[local_i + w * ROW_TILE_SIZE][k] * tile_b_value;
                }
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int w = 0; w < WORK_PER_ITEM; w++) {
        if (i + w * ROW_TILE_SIZE < M && j < N) {
            cs[(i + w * ROW_TILE_SIZE) * N + j] = sums[w];
        }
    }
}