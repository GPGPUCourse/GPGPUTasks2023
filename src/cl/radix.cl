#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#include "../cl_defines.h"
#endif

#line 9


unsigned int get_radix_value(const unsigned int value,
							 const unsigned int offset) {
	return (value & ((1 << (offset + RADIX_BITS)) - (1 << offset))) >> offset;
}


// scan - in-place prefix sum.
// Reduce - pre-calculate every (offset*2)-th element
// offset: must be a power of 2; offset * 2 <= limit
__kernel void scan_reduce_global(__global unsigned int *array,
						  const unsigned int limit,
						  const unsigned int offset)
{
	// TODO: rewrite for many bins
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
__kernel void scan_down_sweep_global(__global unsigned int *array,
							  const unsigned int limit,
							  const unsigned int offset)
{
	// TODO: rewrite for many bins
	const unsigned int gid = get_global_id(0);
//	const unsigned int limit = limit_w * limit_h;
	unsigned long index_prev = (gid + 1) * offset * 2 - 1;
	unsigned long index_cur = index_prev + offset;
	if (index_cur >= limit)
		return;
	array[index_cur] += array[index_prev];
}

__kernel void matrix_transpose(__global const float *input,
                               __global float *output,
                               const int M,
                               const int K)
{
    const unsigned int lid_x = get_local_id(0);
    const unsigned int lid_y = get_local_id(1);
    const unsigned int gid_x = get_global_id(0);
    const unsigned int gid_y = get_global_id(1);
    const unsigned int wid_x = get_group_id(0);
    const unsigned int wid_y = get_group_id(1);
	
	const unsigned int size = TRANSPOSE_WORKGROUP_SIZE;

    __local float buffer[size * size];
	if (gid_x < M && gid_y < K)
    	buffer[lid_y * size + (lid_x + lid_y) % size] = input[gid_y * M + gid_x];

    barrier(CLK_LOCAL_MEM_FENCE);
    const unsigned int index_x = wid_y * size + lid_x;
    const unsigned int index_y = wid_x * size + lid_y;

	if (index_y < M && index_x < K)
   		output[index_y * K + index_x] = buffer[lid_x * size + (lid_y + lid_x) % size];
}

__kernel void count(__global const unsigned int *array,
					__global unsigned int *buffer,
					const int n,
					const int bit_offset) {
	// buffer is `bins` wide, 'chunks` tall
	const unsigned int lid = get_local_id(0);
	const unsigned int bins_num = 1 << RADIX_BITS;
	const unsigned int chunk = get_group_id(0);
	const unsigned int chunk_size = WORKGROUP_SIZE;
	const unsigned int iterations = WORK_PER_THREAD;

	unsigned int count = 0;
	__local unsigned int bins[chunk_size];

	//						 lid % bins_num
	const unsigned int bin = lid & (bins_num - 1);
	// 							lid / bins_num
	const unsigned int offset = lid >> RADIX_BITS;

	for (int i = 0; i < bins_num; ++i) {
		int idx = chunk * chunk_size + i * iterations + offset;
		if (idx > n)
			break;
		if (get_radix_value(array[idx], bit_offset) == bin)
			++count;
	}
	bins[lid] = count;
	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid < bins_num) {
		count = 0;
		for (int i = 0; i < iterations; ++i) {
			count += bins[lid + bins_num * i];
		}
		buffer[bins_num * chunk + lid] = count;
	}
}


__kernel void radix(__global unsigned int *as) {
#if WITH_LOCAL_SORT
	// sort every chunk locally
#endif
    // TODO
}
