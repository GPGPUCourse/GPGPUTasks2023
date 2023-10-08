__kernel void matrix_multiplication_naive(__global const float *lmat, __global const float *rmat, __global float *fmat,
                                          const unsigned int lnrow, const unsigned int ncommon,
                                          const unsigned int rncol) {
    const int fcol = get_global_id(0);
    const int frow = get_global_id(1);

    float fval = 0.0f;

    int lrow = frow;
    int rcol = fcol;
    int lncol = ncommon;
    int rnrow = ncommon;

    for (unsigned int i = 0; i < ncommon; i++) {
        int lcol = i;
        int rrow = i;
        fval += lmat[(lrow * lncol) + lcol] * rmat[(rrow * rncol) + rcol];
    }

    fmat[(frow * rncol) + fcol] = fval;
}

#define TILE_SIZE 16
__kernel void matrix_multiplication_local_memory(__global const float *lmat, __global const float *rmat,
                                                 __global float *fmat, const unsigned int lnrow,
                                                 const unsigned int ncommon, const unsigned int rncol) {
    const int fcol = get_global_id(0);
    const int frow = get_global_id(1);

    int local_col = get_local_id(0);
    int local_row = get_local_id(1);

    __local float ltile[TILE_SIZE][TILE_SIZE];
    __local float rtile[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < ncommon; tileK++) {
        int kstart = tileK * TILE_SIZE;
        ltile[local_row][local_col] = lmat[(frow * ncommon) + (kstart + local_col)];
        rtile[local_row][local_col] = rmat[((kstart + local_row) * rncol) + fcol];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; k++)
            sum += ltile[local_row][k] * rtile[k][local_col];

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    fmat[(frow * rncol) + fcol] = sum;
}

#define WORK_PER_THREAD 4
#define RTS (TILE_SIZE / WORK_PER_THREAD)
__kernel void matrix_multiplication_more_work_per_thread(__global const float *lmat, __global const float *rmat,
                                                         __global float *fmat, const unsigned int lnrow,
                                                         const unsigned int ncommon, const unsigned int rncol) {
    unsigned int local_col = get_local_id(0);
    unsigned int local_row = get_local_id(1);

    unsigned int fcol = get_global_id(0);
    unsigned int frow = get_group_id(1) * TILE_SIZE + local_row;

    __local float ltile[TILE_SIZE][TILE_SIZE];
    __local float rtile[TILE_SIZE][TILE_SIZE];

    float sum[WORK_PER_THREAD];
    for (unsigned int i = 0; i < WORK_PER_THREAD; i++) {
        sum[i] = 0.0f;
    }

    for (unsigned int kstart = 0; kstart < ncommon; kstart+=TILE_SIZE) {

        for (unsigned int i = 0; i < WORK_PER_THREAD; i++) {
            unsigned int frow_actual = frow + i * RTS;
            ltile[local_row + i * RTS][local_col] =
                    (fcol < rncol && frow_actual < lnrow) ? lmat[frow_actual * ncommon + kstart + local_col] : 0;
            rtile[local_row + i * RTS][local_col] =
                    (fcol < rncol && frow_actual < lnrow) ? rmat[(kstart + local_row + i * RTS) * rncol + fcol] : 0;
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (unsigned int k = 0; k < TILE_SIZE; k++) {
            float tmp = rtile[k][local_col];
            for (unsigned int i = 0; i < WORK_PER_THREAD; i++) {
                sum[i] += ltile[local_row + i * RTS][k] * tmp;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (unsigned int i = 0; i < WORK_PER_THREAD; i++) {
        unsigned int frown = frow + i * RTS;
        if (fcol < rncol && frown < lnrow) {
            fmat[frown * rncol + fcol] = sum[i];
        }
    }
}
