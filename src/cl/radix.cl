#define WG_SIZE 128

#define BLOCK_BITS 4
#define BLOCK_VALUES (1 << BLOCK_BITS)

#define DIGIT(value, shift) \
    (((value) >> shift) & (BLOCK_VALUES - 1))

/// \pre work group size is at least BLOCK_VALUES
__kernel void radix_count_table(
    int shift,
    __global const unsigned *input,
    __global unsigned *count_table,
    __global unsigned *aux
) {
    int global_id = get_global_id(0);
    unsigned this = input[global_id];
    unsigned this_digit = DIGIT(this, shift);
    int local_id = get_local_id(0);
    __local int counts[BLOCK_VALUES];
    if (local_id < BLOCK_VALUES) {
        counts[local_id] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_inc(counts + this_digit);
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id < BLOCK_VALUES) {
        count_table[get_group_id(0) * BLOCK_VALUES + local_id] = counts[local_id];
    }
    // sort locally
    __local unsigned group[2][WG_SIZE];
    group[0][local_id] = this;
    barrier(CLK_LOCAL_MEM_FENCE);
    int from = 0;
    for (int blockLength = 1; blockLength < WG_SIZE; blockLength *= 2, from ^= 1) {
        int blockBegin = local_id * blockLength * 2;
        if (blockBegin < WG_SIZE) {
            int i = blockBegin;
            int j = blockBegin + blockLength;
            int k = blockBegin;
            while (i < blockBegin + blockLength && j < blockBegin + 2 * blockLength) {
                if (DIGIT(group[from][i], shift) <= DIGIT(group[from][j], shift)) {
                    group[from ^ 1][k++] = group[from][i++];
                    continue;
                }
                group[from ^ 1][k++] = group[from][j++];
            }
            while (i < blockBegin + blockLength)
                group[from ^ 1][k++] = group[from][i++];
            while (j < blockBegin + 2 * blockLength)
                group[from ^ 1][k++] = group[from][j++];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    aux[global_id] = group[from][local_id];
}

__kernel void radix_sort(
    int shift,
    __global unsigned *output,
    __global unsigned *head_table_t,
    __global const unsigned *counts_table,
    __global unsigned *aux,
    int n
) {
    int global_id = get_global_id(0);
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    unsigned this = aux[global_id];
    unsigned this_digit = DIGIT(this, shift);
    __local int counts[BLOCK_VALUES];
    for (int index = local_id; index < BLOCK_VALUES; index += WG_SIZE) {
        counts[index] = counts_table[group_id * BLOCK_VALUES + index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if (local_id == 0) {
        for (int index = 0; index + 1 < BLOCK_VALUES; ++index)
            counts[index + 1] += counts[index];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int output_index = local_id;
    int head_index = this_digit * n / WG_SIZE + group_id;
    output_index += head_table_t[head_index];
    output_index -= counts[this_digit];
    output[output_index] = this;
}
