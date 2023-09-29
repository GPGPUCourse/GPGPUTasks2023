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
                                                 unsigned int M, unsigned int K,
                                                 unsigned int N) {
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
    }
    cs_gpu[j * N + i] = result;
}

// #define TILE_SIZE 32
// __kernel void matrix_multiplication_local_memory(__global float *as_gpu, __global float *bs_gpu, __global float *cs_gpu,
//                                                  unsigned int M, unsigned int K, unsigned int N) {
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
//         for (int workgroup_tile_i = 0; workgroup_tile_i < TILE_SIZE; workgroup_tile_i += local_size_x) {
//             for (int workgroup_tile_j = 0; workgroup_tile_j < TILE_SIZE; workgroup_tile_j += local_size_y) {
//                 tileA[workgroup_tile_j + local_j][workgroup_tile_i + local_i] =
//                         as_gpu[j * K + (tileK * TILE_SIZE + workgroup_tile_i)];
//                 tileB[workgroup_tile_j + local_j][workgroup_tile_i + local_i] =
//                         bs_gpu[(tileK * TILE_SIZE + workgroup_tile_j) * N + i];
//             }
//         }
//         barrier(CLK_LOCAL_MEM_FENCE);
//         for (int k = 0; k < TILE_SIZE; ++k) {
//             for (int workgroup_tile_i = 0; workgroup_tile_i < TILE_SIZE; workgroup_tile_i += local_size_x) {
//                 for (int workgroup_tile_j = 0; workgroup_tile_j < TILE_SIZE; workgroup_tile_j += local_size_y) {
//                     result += tileA[workgroup_tile_j + local_j][k] * tileB[k][workgroup_tile_i + local_i];
//                 }
//             }
//         }
//     }
//     cs_gpu[j * N + i] = result;
// }