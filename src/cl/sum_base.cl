#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
    #include <libgpu/opencl/cl/common.cl>
#endif

#line 6

__kernel void sum(__global int* arr, __global int* sum){
    const unsigned int gid = get_global_id(0);
    if(gid == 0)
        *sum = 0;
    atomic_add(sum, arr[gid]);
}
