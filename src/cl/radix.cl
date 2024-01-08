inline bool
comp(unsigned int e1, unsigned int e2, bool weak)
{
    return weak ? e1 <= e2 : e1 < e2;
}

__kernel void
merge(
    const __global unsigned int *input,
    __global unsigned int *output,
    unsigned int n,
    unsigned int block_size,
    const unsigned int offset,
    const unsigned int mask)
{
    const int i = get_global_id(0);

    if(i >= n) return;

    int block_index = i / block_size;
    int index_inside_block = i % block_size;
    bool from_left_side = (index_inside_block < (block_size / 2));
    unsigned int my_value = input[i];

    int ind = block_index * block_size + from_left_side * (block_size / 2);
    const __global unsigned int *another_block = input + ind;

    int l = 0, r = l + min(block_size / 2, n - ind);

    while(l < r) {
        int mid = (l + r) / 2;
        if(comp(((my_value >> offset) & mask), ((*(another_block + mid) >> offset) & mask), from_left_side))
            r = mid;
        else
            l = mid + 1;
    }

    __global unsigned int *output_ptr = output + block_index * block_size + index_inside_block + r
        - (!from_left_side) * (block_size / 2);
    *output_ptr = my_value;
}

__kernel void
count(
    __global const unsigned int *as,
    __global unsigned int *cs,
    const unsigned int n,
    const unsigned int offset,
    const unsigned int mask)
{
    const unsigned i = get_global_id(0);
    const unsigned local_id = get_local_id(0);
    const unsigned group_id = get_group_id(0);

    __local unsigned int cnt[K];
    int work_per_item = (K + WG - 1) / WG;
    for(int j = 0; j < work_per_item; j++) {
        if(local_id * work_per_item + j >= K) break;
        cnt[local_id * work_per_item + j] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    if(i < n) {
        unsigned int elem = (as[i] >> offset) & mask;
        atomic_add(cnt + elem, 1);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int j = 0; j < work_per_item; j++) {
        int ind = local_id * work_per_item + j;
        if(ind >= K) break;
        cs[group_id * K + ind] = cnt[ind];
    }
}

__kernel void fill_zeros(
    __global unsigned int *as,
    unsigned int n)
{
    int i = get_global_id(0);
    if(i > n)
        return;
    as[i] = 0;
}

__kernel void update_blocks(
    __global unsigned int *bs,
    unsigned int block_size,
    unsigned int n)
{
    int i = get_global_id(0);
    if(i * block_size + block_size / 2 < n)
        bs[i * block_size] += bs[i * block_size + block_size / 2];
}

__kernel void prefix_sum(
    __global unsigned int *as,
    __global unsigned int *bs,
    unsigned int block_size,
    unsigned int n)
{
    int i = get_global_id(0);
    if(i < n && ((i + 1)&block_size) != 0) {
        as[i] += bs[((i + 1) / block_size - 1) * block_size];
    }
}

__kernel void
matrix_transpose(
    const __global unsigned int *a,
    __global unsigned int *at,
    unsigned int m,
    unsigned int k)
{
    int j = get_global_id(0);
    int i = get_global_id(1);
    __local float tile[TILE_SIZE][TILE_SIZE];
    int local_j = get_local_id(0);
    int local_i = get_local_id(1);
    tile[local_j][(local_i + local_j) % TILE_SIZE] = a[i * k + j]; //циклически сдвигаем каждую строку на номер строки вправо
    barrier(CLK_LOCAL_MEM_FENCE);
    at[(j - j % TILE_SIZE + local_i) * m + (i - i % TILE_SIZE + local_j)] = tile[local_i][(local_i + local_j) % TILE_SIZE];
}

__kernel void
sub_prev_row(
    const __global unsigned int *as,
    __global unsigned int *bs,
    unsigned int n,
    unsigned int m)
{
    int j = get_global_id(0) % m;
    int i = get_global_id(0) / m;
    if(i < n) {
        bs[i * m + j] = as[i * m + j];
        if(i > 0) bs[i * m + j] -= as[i * m - 1];
    }
}

__kernel void
radix(
    const __global unsigned int *as,
    __global unsigned int *bs,
    const __global unsigned int *pf,
    const __global unsigned int *pf_local,
    const unsigned int n,
    const unsigned int offset,
    const unsigned int mask)
{
    const unsigned i = get_global_id(0);
    const unsigned local_id = get_local_id(0);
    const unsigned group_id = get_group_id(0);
    unsigned int groups_cnt = (n + WG - 1) / WG;

    if(i < n) {
        unsigned int elem = (as[i] >> offset) & mask;
        int ind = 0;
        if(elem > 0 || group_id > 0)
            ind += pf[groups_cnt * elem + group_id - 1];
        ind += local_id;
        if(elem > 0)
            ind -= pf_local[K * group_id + (elem - 1)];
        bs[ind] = as[i];
    }
}
