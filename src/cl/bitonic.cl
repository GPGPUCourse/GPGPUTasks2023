#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void bitonic(__global float *as,
					  unsigned int size,
					  unsigned int iter_size,
					  unsigned int iter_stage) {
	unsigned int gid = get_global_id(0);
	const int direction = (gid / iter_size) % 2 ? 0 : 1;
	const unsigned int start = (gid / iter_stage) * iter_stage * 2 + gid % iter_stage;
	const unsigned int end = start + iter_stage;

	if (end < size)
	{
		if ((as[start] > as[end]) == direction) {
			float tmp = as[start];
			as[start] = as[end];
			as[end] = tmp;
		}
	}
}

__kernel void bitonic_fast(__global float *as,
					  unsigned int size,
					  unsigned int iter_size_pow_2,
					  unsigned int iter_stage_pow_2) {
	unsigned int gid = get_global_id(0);
	const int direction = ((gid >> iter_size_pow_2) & 1) ? 0 : 1;
	const unsigned int start = ((gid >> iter_stage_pow_2) << (iter_stage_pow_2 + 1)) + (gid & ((1 << iter_stage_pow_2) - 1));
	const unsigned int end = start + (1 << iter_stage_pow_2);

	if (end < size)
	{
		if ((as[start] > as[end]) == direction) {
			float tmp = as[start];
			as[start] = as[end];
			as[end] = tmp;
		}
	}
}