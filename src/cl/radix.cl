#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6


#define WG_SIZE 128
#define MAX_NUM (1 << 4)

__kernel void prefix_sum(__global const unsigned int *in,
                         __global unsigned int *out,
                         unsigned int length,
                         unsigned int level) {
    const unsigned int i = get_global_id(0);
    if (i < length) {
        if (((i + 1) >> level) & 1) {
            out[i] += in[((i + 1) >> level) - 1];
        }
    }
}

__kernel void prefix_sum_other(__global const unsigned int *in,
                               __global unsigned int *out,
                               unsigned int length) {
    const unsigned int i = get_global_id(0);
    if (i < length) {
        out[i] = in[2 * i] + in[2 * i + 1];
    }
}

__kernel void
radix_count(__global unsigned int *in, __global unsigned int *out, unsigned int size, unsigned int byte_num,
            unsigned int byte_cnt) {
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);

    __local unsigned int lds_buffer[WG_SIZE];
    __local unsigned int res_buffer[MAX_NUM];

    lds_buffer[local_id] = in[global_id];

    if (local_id < (1 << byte_cnt)) {
        res_buffer[local_id] = 0;
    }


    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id == 0) {
        for (unsigned int i = 0; i < WG_SIZE; i++) {
            unsigned int j = (lds_buffer[i] >> byte_num) & ((1 << byte_cnt) - 1);
            res_buffer[j] += 1;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    if (local_id < (1 << byte_cnt)) {
        out[local_id * (size / WG_SIZE) + global_id / WG_SIZE] = res_buffer[local_id];
    }
}


__kernel void radix_sort(__global unsigned int *as, __global unsigned int *prefix, __global unsigned int *res_buffer,
                         unsigned int size, unsigned int byte_num, unsigned int byte_cnt) {
    unsigned int global_id = get_global_id(0);
    unsigned int local_id = get_local_id(0);
    unsigned int num_prefix = global_id / WG_SIZE;

    __local unsigned int lds_buffer[WG_SIZE];
    __local unsigned int lds_counters[MAX_NUM];
    __local unsigned int lds_prev_counts[MAX_NUM];
    __local unsigned int lds_local_res[WG_SIZE];

    lds_buffer[local_id] = as[global_id];

    if (local_id < (1 << byte_cnt)) {
        if (num_prefix == 0) {
            if (local_id == 0)
                lds_prev_counts[local_id] = 0;
            else
                lds_prev_counts[local_id] = prefix[local_id * (size / WG_SIZE) - 1];
        } else {
            lds_prev_counts[local_id] = prefix[local_id * (size / WG_SIZE) + num_prefix - 1];
        }
        lds_counters[local_id] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);


    if (local_id == 0) {
        for (unsigned int i = 0; i < WG_SIZE; i++) {
            unsigned int j = (lds_buffer[i] >> byte_num) & ((1 << byte_cnt) - 1);
            lds_local_res[i] = lds_prev_counts[j] + lds_counters[j];
            lds_counters[j] += 1;
        }
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    res_buffer[lds_local_res[local_id]] = lds_buffer[local_id];
}