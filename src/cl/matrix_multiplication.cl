#define ARGS                 \
    __global const float *A, \
    __global const float *B, \
    __global float *C,       \
    unsigned M,              \
    unsigned K,              \
    unsigned N

__kernel void matrix_multiplication_naive(ARGS) {
    int i = get_global_id(1);
    int j = get_global_id(0);
    float sum = 0;
    for (int r = 0; r < K; ++r)
        sum += A[i * K + r] * B[r * N + j];
    C[i * N + j] = sum;
}

#define TILE_LENGTH 32
/// \pre M, N, K must be divisible by TILE_LENGTH
__kernel void matrix_multiplication_localTile(ARGS) {
    int global_i = get_global_id(1);
    int global_j = get_global_id(0);
    int local_i = get_local_id(1);
    int local_j = get_local_id(0);

    __local float sum[TILE_LENGTH][TILE_LENGTH];
    sum[local_i][local_j] = 0.0f;

    __local float local_A[TILE_LENGTH][TILE_LENGTH];
    __local float local_B[TILE_LENGTH][TILE_LENGTH];
    for (int offset = 0; offset < K; offset += TILE_LENGTH) {
        // Load A, B tiles into local memory
        if (!local_i) {
            for (int local_r = 0; local_r < TILE_LENGTH; ++local_r) {
                local_A[local_r][local_j] = A[(global_i + local_r) * K + offset + local_j];
            }
            for (int local_r = 0; local_r < TILE_LENGTH; ++local_r) {
                local_B[local_r][local_j] = B[(offset + local_r) * N + global_j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply local matrices
        float this_sum = 0.0f;
        for (int local_r = 0; local_r < TILE_LENGTH; ++local_r)
            this_sum += local_A[local_i][local_r] * local_B[local_r][local_j];
        sum[local_i][local_j] += this_sum;
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    C[global_i * N + global_j] = sum[local_i][local_j];
}

#define WORK_THREAD 8
/// \pre M, N, K must be divisible by TILE_LENGTH
__kernel void matrix_multiplication_localTileMoreWorkPerThread(ARGS) {
    int global_i = get_global_id(1);
    global_i *= WORK_THREAD;
    int global_j = get_global_id(0);
    int local_i = get_local_id(1);
    local_i *= WORK_THREAD;
    int local_j = get_local_id(0);

    __local float sum[TILE_LENGTH][TILE_LENGTH];
    for (int w = 0; w < WORK_THREAD; ++w)
        sum[local_i + w][local_j] = 0.0f;

    __local float local_A[TILE_LENGTH][TILE_LENGTH];
    __local float local_B[TILE_LENGTH][TILE_LENGTH];
    for (int offset = 0; offset < K; offset += TILE_LENGTH) {
        // Load A, B tiles into local memory
        if (!local_i) {
            for (int local_r = 0; local_r < TILE_LENGTH; ++local_r) {
                local_A[local_r][local_j] = A[(global_i + local_r) * K + offset + local_j];
            }
            for (int local_r = 0; local_r < TILE_LENGTH; ++local_r) {
                local_B[local_r][local_j] = B[(offset + local_r) * N + global_j];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Multiply local matrices
        for (int local_r = 0; local_r < TILE_LENGTH; ++local_r) {
            float rhs = local_B[local_r][local_j];
            for (int w = 0; w < WORK_THREAD; ++w)
                sum[local_i + w][local_j] += local_A[local_i + w][local_r] * rhs;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int w = 0; w < WORK_THREAD; ++w)
        C[(global_i + w) * N + global_j] = sum[local_i + w][local_j];
}
