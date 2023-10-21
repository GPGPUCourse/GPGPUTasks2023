#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6


__kernel void bitonic(__global float *as, const int block_size, int offset) {
    int id = get_global_id(0);
    int am_i_blue = ((id / block_size) % 2 == 0);
    int inner_id = id % (offset * 2);
    if (inner_id >= offset)
        return;
    if ((am_i_blue && as[id + offset] < as[id]) || (!am_i_blue && as[id + offset] > as[id])) {
        float temp = as[id + offset];
        as[id + offset] = as[id];
        as[id] = temp;
    }
}
