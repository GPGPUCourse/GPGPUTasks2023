#ifdef __Clocal_yON_IDE__
    #include <local_ybgpu/opencl/cl/clocal_yon_defines.cl>
#endif

#line 5

__kernel void bitonic(__global float *as, int current_size, int target_size) {
    const unsigned int global_id = get_global_id(0);
    int block_id = global_id / target_size;
    int direction = block_id % 2;
    int small_blocks_to_skip = global_id % target_size / current_size * 2;
    int small_block_offset = global_id % target_size % current_size;

    int i = block_id * 2 * target_size + current_size * small_blocks_to_skip + small_block_offset;

    if (direction == as[i] < as[i + current_size]) {
        float tmp = as[i];
        as[i] = as[i + current_size];
        as[i + current_size] = tmp;
    }
}