__kernel void matrix_multiplication_naive(__global float *as_gpu, __global float *bs_gpu, __global float *cs_gpu,
                                          unsigned int result_x_size, unsigned int common_size,
                                          unsigned int result_y_size) {
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    float result = 0;
    for (int c = 0; c < common_size; ++c) {
        result += as_gpu[j * common_size + c] * bs_gpu[c * result_y_size + i];
    }
    cs_gpu[j * result_y_size + i] = result;
}

#define TILE_SIZE 16
__kernel void matrix_multiplication_local_memory(__global float *as_gpu, __global float *bs_gpu, __global float *cs_gpu,
                                                 unsigned int M, unsigned int K, unsigned int N) {
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float result = 0;
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        tileA[local_j][local_i] = as_gpu[j * K + (tileK * TILE_SIZE + local_i)];
        tileB[local_j][local_i] = bs_gpu[(tileK * TILE_SIZE + local_j) * N + i];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int k = 0; k < TILE_SIZE; ++k) {
            result += tileA[local_j][k] * tileB[k][local_i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    cs_gpu[j * N + i] = result;
}


#define TILE_SIZE 16
#define WORK_PER_THREAD 1
#define REDUCED_TILE_SIZE (TILE_SIZE / WORK_PER_THREAD)
__kernel void matrix_multiplication_local_memory_more_work_per_thread(__global float *as_gpu, __global float *bs_gpu,
                                                                      __global float *cs_gpu, unsigned int M,
                                                                      unsigned int K, unsigned int N) {
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);

    const unsigned int local_i = get_local_id(0);
    const unsigned int local_j = get_local_id(1);

    __local float tileA[TILE_SIZE][TILE_SIZE];
    __local float tileB[TILE_SIZE][TILE_SIZE];

    float result[WORK_PER_THREAD];
    for (int w = 0; w < WORK_PER_THREAD; ++w) {
        result[w] = 0.0f;
    }

    //
    for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
        for (int w = 0; w < WORK_PER_THREAD; ++w) {
            tileA[local_j + w * REDUCED_TILE_SIZE][local_i] =
                    as_gpu[(j + w * REDUCED_TILE_SIZE) * K + (tileK * TILE_SIZE + local_i)];
            tileB[local_j + w * REDUCED_TILE_SIZE][local_i] =
                    bs_gpu[(tileK * TILE_SIZE + local_j + w * REDUCED_TILE_SIZE) * N + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < TILE_SIZE; ++k) {
            for (int w = 0; w < WORK_PER_THREAD; ++w) {
                result[w] += tileA[local_j][k] * tileB[k][local_i];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // for (int k = 0; k < TILE_SIZE; ++k) {
        //     result[w] += tileA[local_j][k] * tileB[k][local_i];
        // }
        // barrier(CLK_LOCAL_MEM_FENCE);
    }
    //

    for (int w = 0; w < WORK_PER_THREAD; w++) {
        cs_gpu[(j + w * REDUCED_TILE_SIZE) * N + i] = result[w];
    }
}


// #define TS 32
// #define WPT 8
// #define RTS 4
// __kernel void matrix_multiplication_local_memory_more_work_per_thread(__global float *A, __global float *B,
//                                                                       __global float *C, unsigned int M,
//                                                                       unsigned int K, unsigned int N) {
//     // Thread identifiers
//     const int row = get_local_id(0); // Local row ID (max: TS)
//     const int col = get_local_id(1); // Local col ID (max: TS/WPT == RTS)
//     const int globalRow = TS*get_group_id(0) + row; // Row ID of C (0..M)
//     const int globalCol = TS*get_group_id(1) + col; // Col ID of C (0..N)

//     // Local memory to fit a tile of TS*TS elements of A and B
//     __local float Asub[TS][TS];
//     __local float Bsub[TS][TS];

//     // Initialise the accumulation registers
//     float acc[WPT];
//     for (int w=0; w<WPT; w++) {
//         acc[w] = 0.0f;
//     }

//     // Loop over all tiles
//     const int numTiles = K/TS;
//     for (int t=0; t<numTiles; t++) {

//         // Load one tile of A and B into local memory
//         for (int w=0; w<WPT; w++) {
//             const int tiledRow = TS*t + row;
//             const int tiledCol = TS*t + col;
//             Asub[col + w*RTS][row] = A[(tiledCol + w*RTS)*M + globalRow];
//             Bsub[col + w*RTS][row] = B[(globalCol + w*RTS)*K + tiledRow];
//         }

//         // Synchronise to make sure the tile is loaded
//         barrier(CLK_LOCAL_MEM_FENCE);

//         // Perform the computation for a single tile
//         for (int k=0; k<TS; k++) {
//             for (int w=0; w<WPT; w++) {
//                 acc[w] += Asub[k][row] * Bsub[col + w*RTS][k];
//             }
//         }

//         // Synchronise before loading the next tile
//         barrier(CLK_LOCAL_MEM_FENCE);
//     }

//     // Store the final results in C
//     for (int w=0; w<WPT; w++) {
//         C[(globalCol + w*RTS)*M + globalRow] = acc[w];
//     }
// }

// #define TILE_SIZE 32
// __kernel void matrix_multiplication_local_memory(__global const float *as_gpu, __global const float *bs_gpu,
//                                                  __global float *cs_gpu, const unsigned int M, const unsigned int K,
//                                                  const unsigned int N) {
//     const unsigned int i = get_global_id(0);
//     const unsigned int j = get_global_id(1);

//     const unsigned int local_size_x = get_local_size(0);
//     const unsigned int local_size_y = get_local_size(1);

//     const unsigned int local_i = get_local_id(0);
//     const unsigned int local_j = get_local_id(1);

//     __local float tileA[TILE_SIZE][TILE_SIZE];
//     __local float tileB[TILE_SIZE][TILE_SIZE];

//     float result = 0;
//     for (int tileK = 0; tileK * TILE_SIZE < K; ++tileK) {
//         for (unsigned int cur_i = local_i; cur_i < TILE_SIZE; cur_i += local_size_x) {
//             for (unsigned int cur_j = local_j; cur_j < TILE_SIZE; cur_j += local_size_y) {
//                 if (i == 0 && j == 0) {
//                     printf("%d %d\n", cur_i, cur_j);
//                 }
//                 tileA[cur_j][cur_i] = as_gpu[j * K + (tileK * TILE_SIZE + cur_i)];
//                 barrier(CLK_LOCAL_MEM_FENCE);
//                 tileB[cur_j][cur_i] = bs_gpu[(tileK * TILE_SIZE + cur_j) * N + i];
//                 barrier(CLK_LOCAL_MEM_FENCE);
//             }
//         }
//         barrier(CLK_LOCAL_MEM_FENCE);
//         for (unsigned int cur_i = local_i; cur_i < TILE_SIZE; cur_i += local_size_x) {
//             for (unsigned int cur_j = local_j; cur_j < TILE_SIZE; cur_j += local_size_y) {
//                 for (int k = 0; k < TILE_SIZE; ++k) {
//                     result += tileA[cur_j][k] * tileB[k][cur_i];
//                 }
//                 barrier(CLK_LOCAL_MEM_FENCE);
//             }
//         }
//     }
//     cs_gpu[j * N + i] = result;
// }