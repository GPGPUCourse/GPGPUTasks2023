__kernel void matrix_multiplication_naive(
    __global float const *as, 
    __global float const *bs, 
    __global float       *cs, 
    unsigned int          M ,
    unsigned int          K , 
    unsigned int          N  ) {
    
    int x = get_global_id(1);
    int y = get_global_id(0);
    float sum = 0;
    for (int r = 0; r < K; ++r) {
        sum += as[x * K + r] * bs[r * N + y];
    }
    cs[x * N + y] = sum;
}

#define WORK_GROUP_SIZE 32

__kernel void matrix_multiplication_local_mem(
    __global float const *as, 
    __global float const *bs, 
    __global float       *cs, 
    unsigned int          M ,
    unsigned int          K , 
    unsigned int          N  ) {

    const unsigned int groupX = get_group_id(0);
    const unsigned int groupY = get_group_id(1);
    const unsigned int localX = get_local_id(0);
    const unsigned int localY = get_local_id(1);

    __local float tileA[WORK_GROUP_SIZE * WORK_GROUP_SIZE];
    __local float tileB[WORK_GROUP_SIZE * WORK_GROUP_SIZE];

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int Ax = 0;
    unsigned int Bx = 0;
    unsigned int Ay = 0;
    unsigned int By = 0;
    
    float sum = 0.0f;
    for (int k = 0; k < K; k += WORK_GROUP_SIZE) {
        Ax = k + localX;
        Bx = groupX * WORK_GROUP_SIZE + localX;
        Ay = groupY * WORK_GROUP_SIZE + localY;
        By = k + localY;
        tileA[localY * WORK_GROUP_SIZE + localX] = Ay < M && Ax < K ? as[Ay * K + Ax] : 0;
        tileB[localX * WORK_GROUP_SIZE + localY] = By < K && Bx < N ? bs[By * N + Bx] : 0;
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int shift = 0; shift < WORK_GROUP_SIZE; shift++) {
            sum += tileA[localY * WORK_GROUP_SIZE + shift] * tileB[WORK_GROUP_SIZE * localX + shift];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    Ax = groupX * WORK_GROUP_SIZE + localX;
    Ay = groupY * WORK_GROUP_SIZE + localY;
    if (Ax < N && Ay < M) {
        cs[Ay * N + Ax] = sum;
    }
}


#define THREADS 8

__kernel void matrix_multiplication_more_work_per_thread(
    __global float const *as, 
    __global float const *bs, 
    __global float       *cs,       
    unsigned int          M ,              
    unsigned int          K ,              
    unsigned int          N  ) {

    __local float sum   [WORK_GROUP_SIZE][WORK_GROUP_SIZE];
    __local float localA[WORK_GROUP_SIZE][WORK_GROUP_SIZE];
    __local float localB[WORK_GROUP_SIZE][WORK_GROUP_SIZE];
    
    int globalI = get_global_id(1) * THREADS;
    int globalJ = get_global_id(0);
    int localI = get_local_id(1) * THREADS;
    int localJ = get_local_id(0);

    for (int shift = 0; shift < THREADS; shift++) {
        sum[localI + shift][localJ] = 0.0f;
    }

    for (int offset = 0; offset < K; offset += WORK_GROUP_SIZE) {
        if (!localI) {
            for (int localK = 0; localK < WORK_GROUP_SIZE; localK++) {
                localA[localK][localJ] = as[(globalI + localK) * K + offset + localJ];
            }
            for (int localK = 0; localK < WORK_GROUP_SIZE; localK++) {
                localB[localK][localJ] = bs[(offset + localK) * N + globalJ];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int localK = 0; localK < WORK_GROUP_SIZE; localK++) {
            for (int shift = 0; shift < THREADS; shift++) {
                sum[localI + shift][localJ] += localA[localI + shift][localK] * localB[localK][localJ];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for (int shift = 0; shift < THREADS; shift++) {
        cs[(globalI + shift) * N + globalJ] = sum[localI + shift][localJ];
    }
}