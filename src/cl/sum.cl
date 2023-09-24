#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6


#define WAVE_SIZE 32
__kernel void sum_atomic(__global unsigned int *res, __global const unsigned int *data, const unsigned int n) {
    const unsigned thread_id = get_local_id(0);
    const unsigned wg_size = get_local_size(0);
    const unsigned cu_id = get_group_id(0);
    const unsigned cu_cnt = get_num_groups(0);

    const unsigned work_size = (n + cu_cnt - 1) / cu_cnt;
    const unsigned cu_offset = work_size * cu_id;
    unsigned thread_offset = cu_offset + thread_id;

    const unsigned end_offset = min(n, cu_offset + work_size);

    while (thread_offset < end_offset) {
        int value = data[thread_offset];
        atomic_add(res, value);
        thread_offset += wg_size;
    }
}

__kernel void sum_atomic_iter(__global unsigned int *res, __global const unsigned int *data, const unsigned int n) {
    const unsigned thread_id = get_local_id(0);
    const unsigned wg_size = get_local_size(0);
    const unsigned cu_id = get_group_id(0);
    const unsigned cu_cnt = get_num_groups(0);

    const unsigned work_size = (n + cu_cnt - 1) / cu_cnt;
    const unsigned cu_offset = work_size * cu_id;

    const unsigned work_per_thread = (work_size + wg_size - 1) / wg_size;
    unsigned thread_offset = cu_offset + thread_id * work_per_thread;

    unsigned end_offset = min(n, cu_offset + work_size);
    int sum = 0;
    for (int i = 0; i < work_per_thread; ++i) {
        if (thread_offset + i < end_offset) {
            sum += data[thread_offset + i];
        }
    }

    atomic_add(res, sum);
}

__kernel void sum_atomic_coalesce(__global unsigned int *res, __global const unsigned int *data, const unsigned int n) {
    const unsigned thread_id = get_local_id(0);
    const unsigned wg_size = get_local_size(0);
    const unsigned cu_id = get_group_id(0);
    const unsigned cu_cnt = get_num_groups(0);

    const unsigned work_size = (n + cu_cnt - 1) / cu_cnt;
    const unsigned cu_offset = work_size * cu_id;

    unsigned thread_offset = cu_offset + thread_id;

    unsigned end_offset = min(n, cu_offset + work_size);
    int sum = 0;
    while (thread_offset < end_offset) {
        sum += data[thread_offset];
        thread_offset += wg_size;
    }

    atomic_add(res, sum);
}

__kernel void sum_local(__global unsigned int *res, __global const unsigned int *data, const unsigned int n) {
    const unsigned thread_id = get_local_id(0);
    const unsigned wg_size = get_local_size(0);
    const unsigned wave_id = thread_id / WAVE_SIZE;
    const unsigned lane_id = thread_id % WAVE_SIZE;
    const unsigned cu_id = get_group_id(0);
    const unsigned cu_cnt = get_num_groups(0);

    const unsigned work_size = (n + cu_cnt - 1) / cu_cnt;
    const unsigned cu_offset = work_size * cu_id;

    unsigned thread_offset = cu_offset + thread_id;

    __local unsigned lds_buf[256];
    unsigned end_offset = min(n, cu_offset + work_size);
    int sum = 0;

    while (thread_offset < end_offset) {
        sum += data[thread_offset];
        thread_offset += wg_size;
    }

    lds_buf[thread_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    sum = 0;
    if (wave_id == 0) {
        for (int i = 0; i < 4; ++i) {
            sum += lds_buf[i * WAVE_SIZE + lane_id];
        }
        // there should be a cross-lane reduction
        atomic_add(res, sum);
    }
}

__kernel void sum_tree(__global unsigned int *res, __global const unsigned int *data, const unsigned int n) {
    const unsigned thread_id = get_local_id(0);
    const unsigned wg_size = get_local_size(0);
    const unsigned wave_id = thread_id / WAVE_SIZE;
    const unsigned lane_id = thread_id % WAVE_SIZE;
    const unsigned cu_id = get_group_id(0);
    const unsigned cu_cnt = get_num_groups(0);

    const unsigned work_size = (n + cu_cnt - 1) / cu_cnt;
    const unsigned cu_offset = work_size * cu_id;

    unsigned thread_offset = cu_offset + thread_id;

    __local unsigned lds_buf[256];
    unsigned end_offset = min(n, cu_offset + work_size);
    int sum = 0;

    while (thread_offset < end_offset) {
        sum += data[thread_offset];
        thread_offset += wg_size;
    }

    lds_buf[thread_id] = sum;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (wave_id < 4 && wg_size > 128) {
        sum += lds_buf[(wave_id + 4) * WAVE_SIZE + lane_id];
        lds_buf[wave_id * WAVE_SIZE + lane_id] = sum;
        barrier(CLK_LOCAL_MEM_FENCE);
        if (wave_id < 2) {
            sum += lds_buf[(wave_id + 2) * WAVE_SIZE + lane_id];
            lds_buf[wave_id * WAVE_SIZE + lane_id] = sum;
            barrier(CLK_LOCAL_MEM_FENCE);
            if (wave_id == 0) {
                sum += lds_buf[(wave_id + 1) * WAVE_SIZE + lane_id];
                atomic_add(res, sum);
            }

        }

    } else {
        atomic_add(res, sum);
    }
}

