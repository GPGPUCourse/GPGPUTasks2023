#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6
#define VALUES_PER_WORKITEM 32

__kernel void sum_1(
		__global const unsigned int* numbers,
		__global unsigned int* sum,
		const unsigned int n)
{
	const unsigned int gid = get_global_id(0);
	if (gid >= n)
		return;
	atomic_add(sum, numbers[gid]);
}

__kernel void sum_2(
		__global const unsigned int* numbers,
		__global unsigned int* sum,
		const unsigned int n)
{
	const unsigned int gid = get_global_id(0);
	unsigned int local_sum = 0;

	for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
		int local_index = gid * VALUES_PER_WORKITEM + i;
		if (local_index >= n)
			break;
		local_sum += numbers[local_index];
	}
	atomic_add(sum, local_sum);
}

__kernel void sum_3(
		__global const unsigned int* numbers,
		__global unsigned int* sum,
		const unsigned int n)
{
	const unsigned int lid = get_local_id(0);
	const unsigned int wid = get_group_id(0);
	const unsigned int grs = get_local_size(0);
	unsigned int local_sum = 0;

	for (int i = 0; i < VALUES_PER_WORKITEM; ++i) {
		int local_index = wid * grs * VALUES_PER_WORKITEM + i * grs + lid;
		if (local_index >= n)
			break;
		local_sum += numbers[local_index];
	}
	atomic_add(sum, local_sum);
}
