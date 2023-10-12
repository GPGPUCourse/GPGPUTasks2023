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

    //printf("%d %d\n", la, ra);

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