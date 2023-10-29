__kernel void diag_binsearch(__global const float *input,
                                __global int *split_a,
                                __global int *split_b,
                                unsigned int n,
                                unsigned int block_size)
{
    int index = get_global_id(0);
    int items_per_block = (block_size + WORK_PER_ITEM - 1) / WORK_PER_ITEM + 1;
    int block_index = index / items_per_block;
    int work_index = index % items_per_block;

    if(block_index * block_size + max(0, work_index - 1) * WORK_PER_ITEM >= n) return;

    int la = block_index * block_size;
    int lb = block_index * block_size + block_size / 2;
    int a_size = min((int)block_size / 2, max(0, (int)n - la));
    int b_size = min((int)block_size / 2, max(0, (int)n - lb));
    int diag_index = min(work_index * WORK_PER_ITEM - 1, a_size + b_size - 1);
    int l = max(0, diag_index - b_size + 1) - 1;
    int r = min(a_size - 1, diag_index) + 1;
    
    while(l + 1 < r) {
        int mid = (l + r) / 2;
        if(input[la + mid] <= input[lb + diag_index - mid])
            l = mid;
        else
            r = mid;
    }

    split_a[index] = min(n, block_index * block_size + l + 1);
    split_b[index] = min(n, block_index * block_size + block_size / 2 + diag_index - l); 
}


__kernel void merge(__global const float *input,
                    __global float *output,
                    __global const int *split_a,
                    __global const int *split_b,
                    unsigned int n,
                    unsigned int block_size)
{
    int index = get_global_id(0);
    int items_per_block = (block_size + WORK_PER_ITEM - 1) / WORK_PER_ITEM;
    int block_index = index / items_per_block;
    int work_index = index % items_per_block;

    int target_index = block_index * block_size + work_index * WORK_PER_ITEM;

    if(target_index >= n) return;

    int index_in_split_array = block_index * (items_per_block + 1) + work_index;
    int la = split_a[index_in_split_array];
    int ra = split_a[index_in_split_array + 1];
    int lb = split_b[index_in_split_array];
    int rb = split_b[index_in_split_array + 1];

    float block_a[WORK_PER_ITEM];
    float block_b[WORK_PER_ITEM];

    for(int i = la; i < ra; i++)
        block_a[i - la] = input[i];
    for(int i = lb; i < rb; i++)
        block_b[i - lb] = input[i];

    int i = 0, j = 0;
    while(i < ra - la && j < rb - lb) {
        if(block_a[i] < block_b[j]) {
            output[target_index + i + j] = block_a[i];
            ++i;
        }
        else {
            output[target_index + i + j] = block_b[j];
            ++j;
        }
    }
    while(i < ra - la) {
        output[target_index + i + j] = block_a[i];
        ++i;
    }
    while(j < rb - lb) {
        output[target_index + i + j] = block_b[j];
        ++j;
    }
}

__kernel void merge_smart(__global const float *input,
                    __global float *output,
                    __global const int *split_a,
                    __global const int *split_b,
                    unsigned int n,
                    unsigned int block_size)
{
    int index = get_global_id(0);
    int items_per_block = (block_size + WORK_PER_ITEM - 1) / WORK_PER_ITEM;
    int block_index = get_group_id(0) / items_per_block;
    int work_index = get_group_id(0) % items_per_block;
    int i = get_local_id(0);

    int target_index = block_index * block_size + work_index * WORK_PER_ITEM;

    int index_in_split_array = block_index * (items_per_block + 1) + work_index;
    int la = split_a[index_in_split_array];
    int ra = split_a[index_in_split_array + 1];
    int lb = split_b[index_in_split_array];
    int rb = split_b[index_in_split_array + 1];

    __local float block_a[WORK_PER_ITEM];
    __local float block_b[WORK_PER_ITEM];

    if(target_index + i < n && i < block_size) {
        if(i < ra - la)
            block_a[i] = input[la + i];
        else {
            int j = i - (ra - la);
            block_b[j] = input[lb + j];
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(target_index + i >= n || i >= block_size) return;

    if(i < ra - la) {
            int l = 0, r = rb - lb;
            while(l < r) {
                int mid = (l + r) / 2;
                if(block_b[mid] > block_a[i])
                    r = mid;
                else
                    l = mid + 1;
            }
            output[target_index + i + r] = block_a[i];
    }
    else {
            int j = i - (ra - la);
            int l = 0, r = ra - la;
            while(l < r) {
                int mid = (l + r) / 2;
                if(block_a[mid] >= block_b[j])
                    r = mid;
                else
                    l = mid + 1;
            }
            output[target_index + j + r] = block_b[j];
    }
}