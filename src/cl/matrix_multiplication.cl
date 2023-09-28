#define TILE_SIZE 16
#define THREAD_WORK 16

__kernel void matrix_multiplication_base(__global float *as, __global float *bs, __global float *cs,
                                      unsigned int M, unsigned int K, unsigned int N) {
    size_t g_c = get_global_id(0);
    size_t g_r = get_global_id(1);

    float sum = 0.0f;
    for (size_t k = 0; k < K; k++)
        if (g_c < N && g_r < M)
            sum += as[g_r * K + k] * bs[k * N + g_c];

    if (g_c < N && g_r < M)
        cs[g_r * N + g_c] = sum;
}

__kernel void matrix_multiplication_local(__global float *as, __global float *bs, __global float *cs,
                                          unsigned int M, unsigned int K, unsigned int N) {
    size_t g_c = get_global_id(0);
    size_t g_r = get_global_id(1);
    size_t l_c = get_local_id(0);
    size_t l_r = get_local_id(1);

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;

    for (size_t tile = 0; tile < K; tile += TILE_SIZE) {
        tile_a[l_r][l_c] = 0.0f;
        tile_b[l_r][l_c] = 0.0f;

        if (g_r < M && tile + l_c < N)
            tile_a[l_r][l_c] = as[g_r * K + tile + l_c];

        if (g_c < N && tile + l_r < K)
            tile_b[l_r][l_c] = bs[g_c + (tile + l_r) * N];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t k = 0; k < TILE_SIZE; k++)
            sum += tile_a[l_r][k] * tile_b[k][l_c];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (g_c < N && g_r < M)
        cs[g_r * N + g_c] = sum;
}

__kernel void matrix_multiplication_local_thread(__global float *as, __global float *bs, __global float *cs,
                                      unsigned int M, unsigned int K, unsigned int N) {
    const size_t RTS = TILE_SIZE / THREAD_WORK;

    size_t l_c = get_local_id(0);
    size_t l_r = get_local_id(1);
    size_t g_c = get_group_id(0) * TILE_SIZE + l_c;
    size_t g_r = get_group_id(1) * TILE_SIZE + l_r;

    __local float tile_a[TILE_SIZE][TILE_SIZE];
    __local float tile_b[TILE_SIZE][TILE_SIZE];

    float sum[THREAD_WORK] = {};

    for (size_t tile = 0; tile < K; tile += TILE_SIZE) {
        size_t tiel_c = tile + l_c;
        size_t tiel_r = tile + l_r;

        for (size_t w = 0; w < TILE_SIZE; w += RTS) {
            tile_a[l_r + w][l_c] = 0;
            tile_b[l_r + w][l_c] = 0;

            if (tiel_c < K && g_r + w < M)
                tile_a[l_r + w][l_c] = as[(g_r + w) * K + tiel_c];
            if (g_c < N && tiel_r + w < K)
                tile_b[l_r + w][l_c] = bs[(tiel_r + w) * N + g_c];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (size_t k = 0; k < TILE_SIZE; k++) {
            float tmp = tile_b[k][l_c];
            for (size_t w = 0; w < THREAD_WORK; w++)
                sum[w] += tile_a[l_r + RTS * w][k] * tmp;
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (size_t w = 0; w < THREAD_WORK; w++)
        if ((g_r + RTS * w) < M && g_c < N)
            cs[(g_r + RTS * w) * N + g_c] = sum[w];

}