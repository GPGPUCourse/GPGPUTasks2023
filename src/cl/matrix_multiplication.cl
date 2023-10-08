#ifdef __Clocal_yON_IDE__
    #include <local_ybgpu/opencl/cl/clocal_yon_defines.cl>
#endif

#line 6

__kernel void naive(__global const float *as, __global const float *bs, __global float *cs, unsigned int M,
                    unsigned int K, unsigned int N) {
    const unsigned int global_id = get_global_id(0);
    const unsigned int m = global_id / N;
    const unsigned int n = global_id % N;
    float sum = 0.0;
    for (int i = 0; i < K; i++) {
        sum += as[m * K + i] * bs[i * N + n];
    }

    cs[m * N + n] = sum;
}

#define TILE_SIZE 16
__kernel void local_mem(__global const float *as, __global const float *bs, __global float *cs, unsigned int M,
                        unsigned int K, unsigned int N) {
    const unsigned int group_x = get_group_id(0);
    const unsigned int group_y = get_group_id(1);
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);

    __local float tileA[TILE_SIZE * TILE_SIZE];
    __local float tileB[TILE_SIZE * TILE_SIZE];
    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int indexA_x = 0, indexB_x = 0, indexA_y = 0, indexB_y = 0;
    float sum = 0.0f;
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        indexA_x = tile + local_x;
        indexB_x = group_x * TILE_SIZE + local_x;
        indexA_y = group_y * TILE_SIZE + local_y;
        indexB_y = tile + local_y;
        tileA[local_y * TILE_SIZE + local_x] = (indexA_y < M && indexA_x < K) ? as[indexA_y * K + indexA_x] : 0;
        tileB[local_x * TILE_SIZE + local_y] = (indexB_y < K && indexB_x < N) ? bs[indexB_y * N + indexB_x] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[local_y * TILE_SIZE + k] * tileB[TILE_SIZE * local_x + k];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    indexA_x = group_x * TILE_SIZE + local_x;
    indexA_y = group_y * TILE_SIZE + local_y;
    if (indexA_x < N && indexA_y < M)
        cs[indexA_y * N + indexA_x] = sum;
}

#define THREAD_WORK 8
#define TILE_SIZE_DIV_TW (TILE_SIZE / THREAD_WORK)
__kernel void more_work_per_thread(__global const float *as, __global const float *bs, __global float *cs,
                                   const unsigned int M, const unsigned int K, const unsigned int N) {
    const unsigned int group_x = get_group_id(0);
    const unsigned int group_y = get_group_id(1);
    const unsigned int local_x = get_local_id(0);
    const unsigned int local_y = get_local_id(1);

    __local float tileA[TILE_SIZE * TILE_SIZE];
    __local float tileB[TILE_SIZE * TILE_SIZE];

    float sum[THREAD_WORK];
    for (int t = 0; t < THREAD_WORK; t++)
        sum[t] = 0.0f;

    unsigned int indexA_x = 0, indexB_x = 0, indexA_y = 0, indexB_y = 0;
    for (int tile = 0; tile < K; tile += TILE_SIZE) {
        for (int offset_y = 0; offset_y < TILE_SIZE; offset_y += TILE_SIZE_DIV_TW) {
            indexA_x = tile + local_x;
            indexA_y = group_y * TILE_SIZE + local_y + offset_y;
            tileA[(local_y + offset_y) * TILE_SIZE + local_x] =
                    (indexA_x < K && indexA_y < M) ? as[indexA_y * K + indexA_x] : 0.0f;

            indexB_x = group_x * TILE_SIZE + local_x;
            indexB_y = tile + local_y + offset_y;
            tileB[(local_y + offset_y) * TILE_SIZE + local_x] =
                    (indexB_x < N && indexB_y < K) ? bs[indexB_y * N + indexB_x] : 0.0f;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            float tmp = tileB[k * TILE_SIZE + local_x];
            for (int worker = 0; worker < THREAD_WORK; worker++) {
                sum[worker] += tmp * tileA[(local_y + worker * TILE_SIZE_DIV_TW) * TILE_SIZE + k];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int worker = 0; worker < THREAD_WORK; worker++) {
        indexA_x = group_x * TILE_SIZE + local_x;
        indexA_y = group_y * TILE_SIZE + local_y + worker * TILE_SIZE_DIV_TW;
        if (indexA_x < N && indexA_y < M)
            cs[indexA_y * N + indexA_x] = sum[worker];
    }
}