__kernel void merge(const __global float *as_gpu, __global float *bs_gpu, const unsigned int n,
                    const unsigned int merge_block_size) {
    const unsigned int i = get_global_id(0);
    if (i >= n) {
        return;
    }
    bs_gpu[i] = as_gpu[i];
}
