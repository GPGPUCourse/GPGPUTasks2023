#line 2

__kernel void sum_atomic(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n)
{
    unsigned int i = get_global_id(0);
    if (i >= n)
    {
        return;
    }
    atomic_add(sum, arr[i]);
}

#define VALUES_PER_WORKITEM 64
__kernel void sum_atomic_cycle(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n)
{
    size_t gid = get_global_id(0);
    unsigned int temp_sum = 0;
    for (size_t i = 0; i < VALUES_PER_WORKITEM; i++)
    {
        size_t pos = gid * VALUES_PER_WORKITEM + i;
        if (pos < n)
        {
            temp_sum += arr[pos];
        }
    }
    atomic_add(sum, temp_sum);
}

__kernel void sum_atomic_cycle_coalesced(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n)
{
    size_t lid = get_local_id(0);
    size_t wid = get_group_id(0);
    size_t local_size = get_local_size(0);

    unsigned int temp_sum = 0;
    for (size_t i = 0; i < VALUES_PER_WORKITEM; i++)
    {
        size_t pos = wid * local_size * VALUES_PER_WORKITEM + i * local_size + lid;
        if (pos < n)
        {
            temp_sum += arr[pos];
        }
    }
    atomic_add(sum, temp_sum);
}

#define WORKGROUP_SIZE 64
__kernel void sum_local(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n)
{
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);

    __local unsigned int local_buf[WORKGROUP_SIZE];
    local_buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0)
    {
        unsigned int temp_sum = 0;
        for (size_t i = 0; i < WORKGROUP_SIZE; i++)
        {
            temp_sum += local_buf[i];
        }
        atomic_add(sum, temp_sum);
    }
}

__kernel void sum_tree_atomic(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n)
{
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);

    __local unsigned int local_buf[WORKGROUP_SIZE];
    local_buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t nvalues = WORKGROUP_SIZE; nvalues > 1; nvalues /= 2)
    {
        if (2 * lid < nvalues)
        {
            unsigned int a = local_buf[lid];
            unsigned int b = local_buf[lid + nvalues / 2];
            local_buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        atomic_add(sum, local_buf[0]);
    }
}

__kernel void sum_tree_array(__global const unsigned int *arr, __global unsigned int *sum, unsigned int n)
{
    size_t gid = get_global_id(0);
    size_t lid = get_local_id(0);
    size_t wid = get_group_id(0);

    __local unsigned int local_buf[WORKGROUP_SIZE];
    local_buf[lid] = gid < n ? arr[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (size_t nvalues = WORKGROUP_SIZE; nvalues > 1; nvalues /= 2)
    {
        if (2 * lid < nvalues)
        {
            unsigned int a = local_buf[lid];
            unsigned int b = local_buf[lid + nvalues / 2];
            local_buf[lid] = a + b;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        sum[wid] = local_buf[0];
    }
}