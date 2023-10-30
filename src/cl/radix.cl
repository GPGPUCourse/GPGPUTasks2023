__kernel void prefix(
  __global unsigned int* as, 
  __global unsigned int* res, 
  unsigned int offset
) {
    unsigned i = get_global_id(0);
    if (i >= offset) {
      res[i] = as[i] + as[i - offset];
    } else {
      res[i] = as[i];
    }
}


#define WORK_X_SIZE 16
#define WORK_Y_SIZE 16
#define WORK_GROUP_SIZE (WORK_X_SIZE * WORK_Y_SIZE)

__kernel void matrix_transpose(
    __global float *input_matrix, 
    __global float *output_matrix, 
    unsigned int K, 
    unsigned int M) {
    const unsigned int localX = get_local_id(0);
    const unsigned int localY = get_local_id(1);
    const unsigned int globalX = get_group_id(0);
    const unsigned int globalY = get_group_id(1);

    __local float buffer[WORK_GROUP_SIZE];

    unsigned int localId = (globalY * WORK_X_SIZE + localY) * K + (globalX * WORK_Y_SIZE + localX);

    if (globalX * WORK_Y_SIZE + localX < K && globalY * WORK_X_SIZE + localY < M) {
        buffer[localY * WORK_X_SIZE + localX] = input_matrix[localId];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int outputId = (globalX * WORK_Y_SIZE  + localY) * M + (globalY * WORK_X_SIZE + localX);

    if (globalX * WORK_Y_SIZE + localY < K && globalY * WORK_X_SIZE + localX < M) {
        output_matrix[outputId] = buffer[localX * WORK_Y_SIZE + localY];
    }

}


#define BITS 4

int get_mask(int x, int shift) {
    return (x >> shift) & ((1 << BITS) - 1);
}

__kernel void radix_cnt(
    __global unsigned int *as, 
    __global unsigned int* res, 
    unsigned int shift) {
    __local unsigned int cnt[(1 << BITS)];

    int group_id = get_group_id(0);
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);

    if (!(local_id >> BITS)) {
        cnt[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    int cnt_index = get_mask(as[global_id], shift);

    atomic_add(&cnt[cnt_index], 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    if (!(local_id >> BITS)) {
        res[group_id * (1 << BITS) + local_id] = cnt[local_id];
    }
}

#define WORK_GROUP_SIZE_RADIX 128

__kernel void radix(
    __global unsigned int *as, 
    __global unsigned int* bs, 
    __global unsigned int* prefix_sum, 
    __global unsigned int* cnt, 
    int shift) {
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int group_cnt = get_num_groups(0);
    int global_id = get_global_id(0);
    
    __local unsigned int las[WORK_GROUP_SIZE_RADIX];

    las[local_id] = get_mask(as[global_id], shift);

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int offset = 0;
    unsigned int local_value = las[local_id];

    for (int i = 0; i < local_id; i++) {
        if (local_value == las[i]) {
            offset++;
        }
    }
    int pref_pos = prefix_sum[group_cnt * local_value + group_id];
    int cur_pos =  cnt[group_id * (1 << BITS) + local_value];
    int pos = pref_pos - cur_pos;
    bs[pos + offset] = as[global_id];
}