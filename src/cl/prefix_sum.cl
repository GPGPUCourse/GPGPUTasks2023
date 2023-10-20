#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

// scan - in-place prefix sum.
// Reduce - pre-calculate every (offset*2)-th element
// offset: must be a power of 2; offset * 2 <= limit
__kernel void scan_reduce(__global unsigned int *array,
						  const unsigned int limit,
						  const unsigned int offset)
{
	const unsigned int gid = get_global_id(0);
	unsigned int index_cur = (gid + 1) * offset * 2 - 1;
	if (index_cur >= limit)
		return;

	unsigned int index_prev = index_cur - offset;
	array[index_cur] += array[index_prev];
}

// scan - in-place prefix sum.
// DownSweep - calculate every other (offset*2)-th element
// offset: must be a power of 2; offset * 2 <= limit
__kernel void scan_down_sweep(__global unsigned int *array,
							  const unsigned int limit,
							  const unsigned int offset)
{
	const unsigned int gid = get_global_id(0);
//	const unsigned int limit = limit_w * limit_h;
	unsigned long index_prev = (gid + 1) * offset * 2 - 1;
	unsigned long index_cur = index_prev + offset;
	if (index_cur >= limit)
		return;
	array[index_cur] += array[index_prev];
}
