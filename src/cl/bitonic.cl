__kernel void bitonic(__global float *as, const unsigned int n, const unsigned int cur_block_size,
                      const unsigned int j) {
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    bool is_reversed = i % (cur_block_size << 1) >= cur_block_size;
    unsigned int cur_i = (i / j) * (j << 1) + (i % j);
    if (cur_i + j < n && (is_reversed && as[cur_i] < as[cur_i + j] || !is_reversed && as[cur_i] > as[cur_i + j])) {
        const float tmp = as[cur_i];
        as[cur_i] = as[cur_i + j];
        as[cur_i + j] = tmp;
    }
}

// #define WORK_GROUP_SIZE 32

// __kernel void bitonic_coalesed(__global float *as, const unsigned int n, const unsigned int cur_block_size,
//                       const unsigned int j) {
//     const unsigned int i = get_global_id(0);
//     const unsigned int local_i = get_local_id(0);
//     if (i >= n) {
//         return;
//     }

//     if (j <= 32) {
//         __local float line[2 * WORK_GROUP_SIZE];
//         unsigned int cur_i = ((i / j) * (j << 1) + (i % j)) % (WORK_GROUP_SIZE * 2);

//         line[local_i] = as[i];
//         line[local_i + WORK_GROUP_SIZE] = as[i + WORK_GROUP_SIZE];
//         barrier(CLK_LOCAL_MEM_FENCE);
//         bool is_reversed = i % (cur_block_size << 1) >= cur_block_size;
//         if (cur_i + j < n &&
//             (is_reversed && line[cur_i] < line[cur_i + j] || !is_reversed && line[cur_i] > line[cur_i + j])) {
//             const float tmp = line[cur_i];
//             line[cur_i] = line[cur_i + j];
//             line[cur_i + j] = tmp;
//         }
//         barrier(CLK_LOCAL_MEM_FENCE);
//         as[i] = line[local_i];
//         as[i + WORK_GROUP_SIZE] = line[local_i + WORK_GROUP_SIZE];
//     } else {
//         __local float line[WORK_GROUP_SIZE];
//         __local float line2[WORK_GROUP_SIZE];
//         unsigned int cur_i = ((i / j) * (j << 1) + (i % j)) % WORK_GROUP_SIZE;

//         line[local_i] = as[i];
//         line2[local_i] = as[i + j];
//         barrier(CLK_LOCAL_MEM_FENCE);
//         bool is_reversed = i % (cur_block_size << 1) >= cur_block_size;
//         if (cur_i + j < n &&
//             (is_reversed && line[cur_i] < line2[cur_i] || !is_reversed && line[cur_i] > line2[cur_i])) {
//             const float tmp = line[cur_i];
//             line[cur_i] = line2[cur_i];
//             line2[cur_i] = tmp;
//         }
//         barrier(CLK_LOCAL_MEM_FENCE);
//         as[i] = line[local_i];
//         as[i + j] = line2[local_i];
//     }
// }
