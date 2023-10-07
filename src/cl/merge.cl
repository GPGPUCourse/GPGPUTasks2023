#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

__kernel void merge(__global float* a, __global float* b, unsigned int n, unsigned int k) {
    int gid = get_global_id(0);

    int batch_id = gid / k;
    int local_id = gid - batch_id * k;

    bool is_left = batch_id % 2 == 0;
    int other_batch_id =  is_left ? batch_id + 1 : batch_id - 1;
    int other_batch_bound = other_batch_id * k;
    int left_bound = is_left ? batch_id * k : (batch_id - 1) * k;

    int start = other_batch_bound;
    int end = other_batch_bound + k;
    while (start < end) {
        int mid = (start + end) / 2;
        if (a[gid] < a[mid] || (is_left && a[gid] == a[mid]))
            end = mid;
        else
            start = mid + 1;
    }

    b[left_bound + local_id + end - other_batch_bound] = a[gid];
}