#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#define WIDTH (4)
#define WIDTH_POWER (1 << WIDTH)
#define get_masked(x, shift) (((x) >> ((shift) * WIDTH)) & (WIDTH_POWER - 1))

__kernel void count(__global const unsigned int *as, __global int *cs, unsigned int shift) {
    unsigned int gid = get_global_id(0);
    unsigned int lid = get_local_id(0);

    __local int counters[WIDTH_POWER];

    if (lid < WIDTH_POWER) {
        counters[lid] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    atomic_add(&counters[get_masked(as[gid], shift)], 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    const int base = get_group_id(0) * WIDTH_POWER;
    if (lid < WIDTH_POWER) {
        cs[base + lid] = counters[lid];
    }
}

__kernel void radix(__global const unsigned int *as, unsigned int shift,
                    __global const unsigned int *prefix_c,
                    __global const unsigned int *prefix_p_t,
                    __global unsigned int *res) {
    unsigned int lid = get_local_id(0);
    unsigned int value = as[get_global_id(0)];
    unsigned int masked = get_masked(value, shift);
    unsigned int group_id = get_group_id(0);
    unsigned int num_groups = get_num_groups(0);

    int p_index = prefix_p_t[masked * num_groups + group_id];
    int delta_c_index = prefix_c[group_id * WIDTH_POWER + masked] - prefix_c[group_id * WIDTH_POWER];

    int res_idx = p_index + lid - delta_c_index;
//    printf("global_id: %d, masked: %d, group_id: %d, num_groups: %d\n"
//           "value: %d (p_index(%d) + lid(%d) - delta_c_index(%d) => res_idx: %d)\n",
//           get_global_id(0), masked, group_id, num_groups,
//           value, p_index, lid, delta_c_index, res_idx);

    res[res_idx] = value;
}

#undef get_masked
