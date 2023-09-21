#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void sum1(__global const unsigned int* a,
                   const unsigned int n,
                   __global unsigned int* sum)
{
    const unsigned int i = get_global_id(0);
    if (i >= n)
        return;
    atomic_add(sum, a[i]);
}

__kernel void sum2(__global const unsigned int* a,
                   const unsigned int n,
                   __global unsigned int* sum)
{
    const unsigned int values_per_workitem = 128;
    const unsigned int i = get_global_id(0);
    unsigned int tmp = 0;

    for (int j = 0; j < values_per_workitem; j++)
    {
        unsigned int ij = i * values_per_workitem + j;
        if (ij < n)
        {
            tmp += a[ij];
        }
    }
    atomic_add(sum, tmp);
}

__kernel void sum3(__global const unsigned int* a,
                   const unsigned int n,
                   __global unsigned int* sum)
{
    const unsigned int values_per_workitem = 128;
    const unsigned int groupId = get_group_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int grs = get_local_size(0);
    unsigned int tmp = 0;

    for (int j = 0; j < values_per_workitem; j++)
    {
        unsigned int ij = groupId * grs * values_per_workitem + j * grs + lid;
        if (ij < n)
        {
            tmp += a[ij];
        }
    }
    atomic_add(sum, tmp);
}

#define WORKGROUP_SIZE 128

__kernel void sum4(__global const unsigned int* a,
                   const unsigned int n,
                   __global unsigned int* sum)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];
    buf[lid] = gid < n ? a[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0)
    {
        unsigned int tmp = 0;
        for (int i = 0; i < WORKGROUP_SIZE; i++)
        {
            tmp += buf[i];
        }
        atomic_add(sum, tmp);
    }
}

__kernel void sum5(__global const unsigned int* a,
                   const unsigned int n,
                   __global unsigned int* sum)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];
    buf[lid] = gid < n ? a[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int level = WORKGROUP_SIZE; level > 1; level >>= 1)
    {
        if (2 * lid < level)
        {
            buf[lid] += buf[lid + level / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        atomic_add(sum, buf[0]);
    }
}

__kernel void sum6(__global const unsigned int* a,
                   const unsigned int n,
                   __global unsigned int* sum)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);
    const unsigned int groupId = get_group_id(0);

    __local unsigned int buf[WORKGROUP_SIZE];
    buf[lid] = gid < n ? a[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int level = WORKGROUP_SIZE; level > 1; level >>= 1)
    {
        if (2 * lid < level)
        {
            buf[lid] += buf[lid + level / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        sum[groupId] = buf[0];
    }
}

__kernel void sum3_global_mem(__global const unsigned int* a,
                   const unsigned int n,
                   __global unsigned int* sum)
{
    const unsigned int gid = get_global_id(0);
    const unsigned int lid = get_local_id(0);

    if (gid < n) atomic_add(&sum[lid], a[gid]);
}
