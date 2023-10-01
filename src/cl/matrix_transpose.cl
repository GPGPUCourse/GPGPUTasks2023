#define size 16
kernel void matrix_transpose(global const float* a, global float* result, unsigned n, unsigned m) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);
    int gy = get_group_id(1);

    local float buffer[size][size + 1]; // +1 for banking
    buffer[ly][lx] = a[y * m + x];

    barrier(CLK_LOCAL_MEM_FENCE);
    int writeY = x - lx + ly;
    int writeX = y - ly + lx;
    result[writeY * n + writeX] = buffer[lx][ly];
}
