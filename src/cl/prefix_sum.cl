#ifdef __Clocal_yON_IDE__
    #include <local_ybgpu/opencl/cl/clocal_yon_defines.cl>
#endif

#line 5

__kernel void calculate_parts(__global unsigned int *part_sums, unsigned int size) {
    const unsigned int id = get_global_id(0);
    part_sums[id * size] += part_sums[id * size + size / 2];
}

__kernel void calculate_prefix_sums(__global unsigned int *result, __global const unsigned int *part_sums,
                                    unsigned int size) {
    const unsigned int id = get_global_id(0) + 1;
    if (id & size)
        result[id] += part_sums[(id - size) / size * size];
}