// vim: syntax=c

#define BLOCK_SIZE 16
#define THREAD_WORK 4

__kernel void matrix_multiplication_naive(
    __global const float * a,
    __global const float * b,
    __global float * c,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    float sum = 0;
    for (unsigned int i = 0; i < K; ++i) {
        sum += a[y * K + i] * b[i * N + x];
    }

    c[y * N + x] = sum;
}

__kernel void matrix_multiplication_block(
    __global const float * a,
    __global const float * b,
    __global float * c,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    const unsigned int gx = get_global_id(0);
    const unsigned int gy = get_global_id(1);

    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    __local float a_buf[BLOCK_SIZE][BLOCK_SIZE + 1];
    __local float b_buf[BLOCK_SIZE][BLOCK_SIZE + 1];

    float sum = 0;
    for (unsigned int i = 0; i < K / BLOCK_SIZE; ++i) {
        a_buf[ly][lx] = a[gy * K + i * BLOCK_SIZE + lx];
        b_buf[ly][lx] = b[(i * BLOCK_SIZE + ly) * N + gx];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int j = 0; j < BLOCK_SIZE; ++j) {
            sum += a_buf[ly][j] * b_buf[j][lx];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[gy * N + gx] = sum;
}

__kernel void matrix_multiplication_many(
    __global const float * a,
    __global const float * b,
    __global float * c,
    unsigned int M,
    unsigned int K,
    unsigned int N
) {
    const unsigned int grx = get_group_id(0) * BLOCK_SIZE;
    const unsigned int gy = get_global_id(1);

    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    const unsigned int bs2tw = BLOCK_SIZE / THREAD_WORK;

    __local float a_buf[BLOCK_SIZE][BLOCK_SIZE + 1];
    __local float b_buf[BLOCK_SIZE][BLOCK_SIZE + 1];

    float sum[THREAD_WORK];
    for (unsigned int i = 0; i < THREAD_WORK; ++i) {
        sum[i] = 0;
    }

    for (unsigned int i = 0; i < K / BLOCK_SIZE; ++i) {
        for (unsigned int j = 0; j < THREAD_WORK; ++j) {
            const unsigned int cur_x = j * bs2tw + lx;

            a_buf[ly][cur_x] = a[gy * K + i * BLOCK_SIZE + cur_x];
            b_buf[ly][cur_x] = b[(i * BLOCK_SIZE + ly) * N + grx + cur_x];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int j = 0; j < BLOCK_SIZE; ++j) {
            const float a_buf_v = a_buf[ly][j];

            for (unsigned int k = 0; k < THREAD_WORK; ++k) {
                sum[k] += a_buf_v * b_buf[j][k * bs2tw + lx];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (unsigned int i = 0; i < THREAD_WORK; ++i) {
        c[gy * N + grx + i * bs2tw + lx] = sum[i];
    }
}
