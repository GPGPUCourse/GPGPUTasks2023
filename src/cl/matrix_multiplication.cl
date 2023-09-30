__kernel void matrix_multiplication_1(__global const float *a,
                                      __global const float *b,
                                      __global float *c,
                                      unsigned int N,
                                      unsigned int M,
                                      unsigned int K)
{
    int y = get_global_id(0);
    int x = get_global_id(1);

    float sum = 0.0f;
    for (int c = 0; c < M; ++c) {
        sum += a[x * M + c] * b[c * K + y];
    }

    c[x * K + y] = sum;
}

__kernel void matrix_multiplication_2(__global const float *a,
                                      __global const float *b,
                                      __global float *c,
                                      unsigned int N,
                                      unsigned int M,
                                      unsigned int K)
{
    int group_y = get_group_id(0);
    int group_x = get_group_id(1);

    int local_y = get_local_id(0);
    int local_x = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum = 0.f;
    for (int tile = 0; tile * TILE_SIZE < M; ++tile) {
        tileA[local_x][local_y] = a[(group_x * TILE_SIZE + local_x) * M + tile * TILE_SIZE + local_y];
        tileB[local_x][local_y] = b[(tile * TILE_SIZE + local_x) * K + group_y * TILE_SIZE + local_y];
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[local_x][k] * tileB[k][local_y];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

    }

    c[(group_x * TILE_SIZE + local_x) * K + group_y * TILE_SIZE + local_y] = sum;
}

__kernel void matrix_multiplication_3(__global const float *a,
                                      __global const float *b,
                                      __global float *c,
                                      unsigned int N,
                                      unsigned int M,
                                      unsigned int K)
{
    int group_y = get_group_id(0);
    int group_x = get_group_id(1);

    int local_y = get_local_id(0);
    int local_x = get_local_id(1);

    int y = group_y * TILE_SIZE + local_y;
    int x = group_x * TILE_SIZE + local_x;

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float sum[THREAD_WORK];
    for (int t = 0; t < THREAD_WORK; ++t) {
        sum[t] = 0.0f;
    }
    for (int tile = 0; tile < M; tile += TILE_SIZE) {
        for (int t = 0; t < TILE_SIZE; t += GROUP_WORK) {
            tileA[local_x + t][local_y] = a[(x + t) * M + tile + local_y];
            tileB[local_x + t][local_y] = b[(tile + local_x + t) * K + y];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            float tmp = tileB[k][local_y];
            for (int t = 0; t < THREAD_WORK; ++t) {
                sum[t] += tileA[local_x + t * GROUP_WORK][k] * tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

    }

    for (int t = 0; t < THREAD_WORK; ++t) {
        c[(x + t * GROUP_WORK) * K + y] = sum[t];
    }
}
