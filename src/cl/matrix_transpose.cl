// vim: syntax=c

#define X_GROUP_SIZE 16
#define Y_GROUP_SIZE 16

#define assert(__cond) if (!(__cond)) { \
    printf("ASSERTION FAILED " __FILE__ ":%d\n", __LINE__); \
    return; \
}

__kernel void matrix_transpose(
    __global const float * a,
    __global float * a_t,
    const unsigned int M,
    const unsigned int K
) {
    assert(get_local_size(0) == X_GROUP_SIZE);
    assert(get_local_size(1) == Y_GROUP_SIZE);

    const unsigned int gx = get_group_id(0) * X_GROUP_SIZE;
    const unsigned int gy = get_group_id(1) * Y_GROUP_SIZE;
    const unsigned int lx = get_local_id(0);
    const unsigned int ly = get_local_id(1);

    __local float buf[Y_GROUP_SIZE][X_GROUP_SIZE + 1];

    if (gx + lx < M && gy + ly < K) {
        buf[ly][lx] = a[(gy + ly) * M + (gx + lx)];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (gx + ly < M && gy + lx < K) {
        a_t[(gx + ly) * K + (gy + lx)] = buf[lx][ly];
    }
}
