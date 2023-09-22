#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void kernel_sum1(__global const unsigned int* as,
                   __global unsigned int* sum,
                   unsigned int n)
{
    const unsigned int index = get_global_id(0);

    if (index >= n)
        return;

    atomic_add(sum, as[index]);
}


__kernel void kernel_sum2(__global const unsigned int* as,
                   __global unsigned int* sum,
                   const unsigned int values_per_workitem,
                   unsigned int n)
{
    const unsigned int gid = get_global_id(0);

    unsigned int res = 0;
    for(int i = 0; i < values_per_workitem; ++i) {
        int index = gid * values_per_workitem + i;
        if (index < n) {
            res += as[index];
        }
    }

    atomic_add(sum, res);
}


__kernel void kernel_sum3(__global const unsigned int* as,
                   __global unsigned int* sum,
                   const unsigned int values_per_workitem,
                   unsigned int n)
{
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);
    const unsigned int gls = get_local_size(0);

    unsigned int res = 0;
    for(int i = 0; i < values_per_workitem; ++i) {
        int index = wid * gls * values_per_workitem + i * gls + lid;
        if (index < n) {
            res += as[index];
        }
    }

    atomic_add(sum, res);
}


#define WORKGROUP_SIZE 128
__kernel void kernel_sum4(__global const unsigned int* as,
                   __global unsigned int* sum,
                   unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? as[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned int res = 0;
        for(int i = 0; i < WORKGROUP_SIZE; ++i) {
            res += buf[i];
        }
        atomic_add(sum, res);
    }
}


__kernel void kernel_sum5(__global const unsigned int* as,
                   __global unsigned int* sum,
                   unsigned int n)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int wid = get_group_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];

    buf[lid] = gid < n ? as[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nValues = WORKGROUP_SIZE; nValues > 1; nValues >>= 1) {
        if ((lid << 1) < nValues) {
            buf[lid] += buf[lid + (nValues >> 1)];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        sum[wid] = buf[0];
    }
}
