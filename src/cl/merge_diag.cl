#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif

__kernel void calc_idx(__global const float *arr, __global uint *idx_a, __global uint *idx_b, uint len,
                       uint sorted_block_len) {
    // Here work groups don't correspond to phase 2
    uint gid = get_global_id(0);
    uint dst_block_len = 2 * sorted_block_len;
    uint dst_id = WORK_GROUP_SIZE * gid;
    uint diag = dst_id % dst_block_len;
    uint dst_block = dst_id / dst_block_len;
    uint dst_block_start = dst_block * dst_block_len;
}

__kernel void merge(__global const float *src, __global float *dst, __global const uint *idx_a,
                    __global const uint *idx_b, uint len, uint sorted_len) {
    __local float local_src_a[WORK_GROUP_SIZE];
    __local float local_src_b[WORK_GROUP_SIZE];
    __local float local_dst[WORK_GROUP_SIZE];
}
