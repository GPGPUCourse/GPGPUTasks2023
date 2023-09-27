#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

__kernel void baseline(__global const float *a, __global const float *b, __global float *c, unsigned N, unsigned M,
                       unsigned K) {
    const int i = get_global_id(1);
    const int j = get_global_id(0);

    float sum = 0.0;
    for (int k = 0; k < M; ++k) {
        sum += a[i * M + k] * b[k * K + j];
    }

    c[i * K + j] = sum;
}

#define TILE_SIZE 16

__kernel void local_mem(__global const float *a, __global const float *b, __global float *c, unsigned N, unsigned M,
                        unsigned K) {
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    const int i = get_global_id(1);
    const int j = get_global_id(0);

    const int li = get_local_id(1);
    const int lj = get_local_id(0);

    const int gi = get_group_id(1);
    const int gj = get_group_id(0);

    float sum = 0.0;
    for (int block = 0; block * TILE_SIZE < M; ++block) {
        tileA[li][lj] = a[(gi * TILE_SIZE + li) * M + (block * TILE_SIZE + lj)];
        tileB[li][lj] = b[(block * TILE_SIZE + li) * M + (gj * TILE_SIZE + lj)];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            sum += tileA[li][k] * tileB[k][lj];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    c[i * K + j] = sum;
}

__kernel void local_mem_more_work(__global const float *a, __global const float *b, __global float *c, unsigned N,
                                  unsigned M, unsigned K) {
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    // Always 0.
    const int li = get_local_id(1);
    // [0..TILE_SIZE - 1].
    const int lj = get_local_id(0);

    const int gi = get_group_id(1);
    const int gj = get_group_id(0);

    // Каждый поток считает свой столбец.
    float sum[TILE_SIZE];
    for (int i = 0; i < TILE_SIZE; ++i)
        sum[i] = 0.0;

    for (int block = 0; block * TILE_SIZE < M; ++block) {
        for (int row = 0; row < TILE_SIZE; ++row) {
            tileA[row][lj] = a[(gi * TILE_SIZE + row) * M + (block * TILE_SIZE + lj)];
            tileB[row][lj] = b[(block * TILE_SIZE + row) * M + (gj * TILE_SIZE + lj)];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int rowInB = 0; rowInB < TILE_SIZE; ++rowInB) {
            float coef = tileB[rowInB][lj];
            for (int rowInA = 0; rowInA < TILE_SIZE; ++rowInA) {
                sum[rowInA] += tileA[rowInA][rowInB] * coef;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Синхронно пишем в ответ по строкам, каждый пишет в одну и ту же строчку.
    for (int row = 0; row < TILE_SIZE; ++row) {
        c[(gi*TILE_SIZE+row)*K+(gj*TILE_SIZE+lj)] = sum[row];
    }
}
