// TODO
__kernel void sum_gpu_atomic(__global const unsigned int *arr,
                             __global unsigned int *sum)
{
    const unsigned int gid = get_global_id(0);
    atomic_add(sum, arr[gid]);
}