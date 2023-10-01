// vim: syntax=c

#define X_WORK_SIZE 8
#define Y_WORK_SIZE 8

__kernel void matrix_transpose(
    __global const float * a,
    __global float * a_t,
    const unsigned int M,
    const unsigned int K
) {
    // __local float buf[Y_WORK_SIZE][X_WORK_SIZE + 1];

    const unsigned int x = get_global_id(0);
    const unsigned int y = get_global_id(1);

    if (x < M && y < K) {
        // buf[y][x] = a[y * M + x];
        // a_t[x * K + y] = buf[y][x];

        const float v = a[y * M + x];
        a_t[x * K + y] = v;
    }
}
