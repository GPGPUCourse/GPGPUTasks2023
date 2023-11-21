__kernel void matrix_multiplication_naive(
    __global float *a,
    __global float *b,
    __global float *c,
    unsigned int m,
    unsigned int k,
    unsigned int n
)
{


    int i = get_global_id(0);
    int j = get_global_id(1);

    if (i >= n || j >= m)
        return;

    float res = 0;
    for (int t = 0; t < k; t++) {
        res += a[j * k + t] * b[t * n + i];
    }

   c[j * n + i] = res;
}

#ifdef TILE_SIZE
__kernel void matrix_multiplication_local_mem(
    __global float *a,
    __global float *b,
    __global float *c,
    unsigned int m,
    unsigned int k,
    unsigned int n
)
{
    int i = get_global_id(0);
    int j = get_global_id(1);

    int local_i = get_local_id(0);
    int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float res = 0;
    for (int tileId = 0; tileId * TILE_SIZE < k; tileId++) {
        int a_row = j;
        int a_col = local_i + tileId * TILE_SIZE;
        int b_row = local_j + tileId * TILE_SIZE;
        int b_col = i;
        tileA[local_j][local_i] = (a_row < m && a_col < k ? a[a_row * k + a_col] : 0);
        tileB[local_j][local_i] = (b_row < k && b_col < n ? b[b_row * k + b_col] : 0);
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int t = 0; t < TILE_SIZE; t++) {
            res += tileA[local_j][t] * tileB[t][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (j < m && i < n)
        c[j * n + i] = res;
}
#endif

#ifdef TILE_SIZE
#ifdef THREAD_WORK
__kernel void matrix_multiplication_much_thread_work(
    __global float *a,
    __global float *b,
    __global float *c,
    unsigned int m,
    unsigned int k,
    unsigned int n
)
{
    int local_col = get_local_id(0);
    int local_row = get_local_id(1);
    local_row *= THREAD_WORK;

    int global_col = get_global_id(0);
    int global_row = get_global_id(1);
    global_row *= THREAD_WORK;

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float res[THREAD_WORK];
    for (int i = 0; i < THREAD_WORK; i++) {
        res[i] = 0;
    }

    for (int tile = 0; tile < k; tile += TILE_SIZE) {

        for (int t = 0; t < THREAD_WORK; t++) {
            tileA[local_row + t][local_col] = a[(global_row + t) * k + tile + local_col];
            tileB[local_row + t][local_col] = b[(local_row + t + tile) * n + global_col];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int v = 0; v < TILE_SIZE; v++) {
            float tmp = tileB[v][local_col];
            for (int t = 0; t < THREAD_WORK; t++) {
                res[t] += tileA[local_row + t][v] * tmp;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int t = 0; t < THREAD_WORK; t++) {
            c[(global_row + t) * n + global_col] = res[t];
        }
    }
}
#endif
#endif