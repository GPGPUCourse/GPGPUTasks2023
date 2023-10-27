__kernel void naive_multiplication(__global const float *a,
                                    __global const float *b,
                                    __global float *c,
                                    unsigned int M,
                                    unsigned int K,
                                    unsigned int N)
{
    int i = get_global_id(0) / N;
    int j = get_global_id(0) % N;
    float sum = 0.f;
    for(int k = 0; k < K; k++)
        sum += a[i * K + k] * b[k * N + j];
    c[i * N + j] = sum;
}

__kernel void local_memory_multiplication_1(__global const float *a,
                                            __global const float *b,
                                            __global float *c,
                                            unsigned int M,
                                            unsigned int K,
                                            unsigned int N)
{
    int i = get_global_id(1);
    int j = get_global_id(0);
    int local_i = get_local_id(1);
    int local_j = get_local_id(0);
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];
    float sum = 0.f;
    int tileI = i / TILE_SIZE;
    int tileJ = j / TILE_SIZE;
    for(int tileK = 0; tileK * TILE_SIZE < K; tileK++) {
        tileA[local_i][local_j] = a[(tileI * TILE_SIZE + local_i) * K + (tileK * TILE_SIZE + local_j)];
        tileB[local_i][local_j] = b[(tileK * TILE_SIZE + local_i) * N + (tileJ * TILE_SIZE + local_j)];
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[local_i][k] * tileB[k][local_j];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    c[i * N + j] = sum;
}

__kernel void local_memory_multiplication_2(__global const float *a,
                                            __global const float *b,
                                            __global float *c,
                                            unsigned int M,
                                            unsigned int K,
                                            unsigned int N)
{
    //int i = get_global_id(1);
    int tileI = get_global_id(1);
    int j = get_global_id(0);
    int local_i = get_local_id(0);
    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];
    float accs[TILE_SIZE];
    for(int k = 0; k < TILE_SIZE; k++) {
        accs[k] = 0.f;
    }
    //int tileI = i / TILE_SIZE;
    int tileJ = j / TILE_SIZE;
    for(int tileK = 0; tileK * TILE_SIZE < K; tileK++) {
        for(int k = 0; k < TILE_SIZE; k++) {
            tileA[k][local_i] = a[(tileI * TILE_SIZE + k) * K + (tileK * TILE_SIZE + local_i)];
            tileB[k][local_i] = b[(tileK * TILE_SIZE + k) * N + (tileJ * TILE_SIZE + local_i)];
            //tileA[local_i][k] = a[(tileI * TILE_SIZE + local_i) * K + (tileK * TILE_SIZE + k)];
            //tileB[local_i][k] = b[(tileK * TILE_SIZE + local_i) * N + (tileJ * TILE_SIZE + k)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        for(int k = 0; k < TILE_SIZE; k++) {
            float tmp = tileB[k][local_i];
            for(int w = 0; w < TILE_SIZE; w++) {
                accs[w] += tileA[w][k] * tmp;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    for(int k = 0; k < TILE_SIZE; k++) {
        c[(tileI * TILE_SIZE + k) * N + (tileJ * TILE_SIZE + local_i)] = accs[k];
    }
}