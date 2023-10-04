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

    int local_row = get_local_id(0);
    int local_col = get_local_id(0);

    __local float ltile[TILE_SIZE][TILE_SIZE];
    __local float rtile[TILE_SIZE][TILE_SIZE];

    float sum = 0.0f;
    for (int tileK = 0; tileK * TILE_SIZE < ncommon; tileK++) {
        ltile[local_row][local_col] = lmat[(frow * ncommon) + ((tileK * TILE_SIZE) + local_col)];
        rtile[local_row][local_col] = rmat[(((tileK * TILE_SIZE) + local_row) * rncol) + fcol];

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE;)
            sum += ltile[local_row][k] * rtile[k][local_col];
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    fmat[(frow * rncol) + fcol] = sum;
}