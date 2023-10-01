kernel void simple(global const float *a, global const float *b,
                  global float *result, unsigned n, unsigned m, unsigned l) {
    int x = get_global_id(0);
    int y = get_global_id(1);

    float sum = 0;
    for (int i = 0; i < m; ++i) sum += a[y * m + i] * b[i * l + x];
    result[y * l + x] = sum;
}

#define size 16
kernel void localmem(global const float *a, global const float *b,
                     global float *result, unsigned n, unsigned m, unsigned l) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    local float la[size][size], lb[size][size];
    float sum = 0;
    for (int k = 0; k < m; k += size) {
        la[ly][lx] = a[y * m + k + lx];
        barrier(CLK_LOCAL_MEM_FENCE);
        lb[ly][lx] = b[(k + ly) * l + x];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < size; ++i) sum += la[ly][i] * lb[i][lx];
    }
    result[y * l + x] = sum;
}

#define w 4
kernel void morework(global const float *a, global const float *b,
                     global float *result, unsigned n, unsigned m, unsigned l) {
    int x = get_global_id(0);
    int y = get_global_id(1);
    int lx = get_local_id(0);
    int ly = get_local_id(1);

    local float la[size][size], lb[size][w * size];
    float sum[w] = {0};
    for (int k = 0; k < m; k += size) {
        la[ly][lx] = a[y * m + k + lx];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < w; ++i) lb[ly][w * lx + i] = b[(k + ly) * l + w * x + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int i = 0; i < size; ++i) 
            for (int j = 0; j < w; ++j)
                sum[j] += la[ly][i] * lb[i][w * lx + j];
    }
    for (int i = 0; i < w; ++i) result[y * l + w * x + i] = sum[i];
}
