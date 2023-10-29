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
	// ...or not?
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


unsigned int find_local_merge_path(const __local float *array,
								   const unsigned int start0,
								   const unsigned int count0,
								   const unsigned int start1,
								   const unsigned int count1,
								   const unsigned int diag,
								   const unsigned int bit_offset) {
	unsigned int l = max(0, (int)diag - (int)count1);
	unsigned int r = min(diag, count0);
	unsigned int m;
	while (l < r) {
		m = (l + r) >> 1;
		if (get_radix_value(array[start0 + m], bit_offset) < get_radix_value(array[start1 + diag - 1 - m], bit_offset)) {
			l = m + 1;
		} else {
			r = m;
		}
	}
	return l;
}

void find_partition_local(const __local float *array,
						  unsigned int interval[4],
						  const unsigned int leftSize,
						  const unsigned int rightSize,
						  const unsigned int sortedSize,
						  const unsigned int bit_offset) {
	// interval:
	//   [0] - start of left
	//   [1] - end of left
	//   [2] - start of right
	//   [3] - end of right
	const unsigned int lid = get_local_id(0);
	interval[0] = (lid / sortedSize) * sortedSize * 2;
	interval[1] = interval[0] + leftSize;
	interval[2] = interval[1];
	interval[3] = interval[2] + rightSize;
//	if (get_global_id(0) == DEBUG_GID)
//		printf("g%d before %d [%u, %u]-[%u, %u]\n", DEBUG_GID, sortedSize, interval[0], interval[1], interval[2], interval[3]);

	unsigned int diag = (lid % sortedSize) * 2;
//	if (get_global_id(0) == DEBUG_GID)
//		printf("g%d diags: %d, %d\n", DEBUG_GID, diag, diag + 2);
	int idx_start = find_local_merge_path(array,
										  interval[0],
										  interval[1] - interval[0],
										  interval[2],
										  interval[3] - interval[2],
										  diag, bit_offset);
	int idx_end = find_local_merge_path(array,
										interval[0],
										interval[1] - interval[0],
										interval[2],
										interval[3] - interval[2],
										diag + 2, bit_offset);
	interval[1] = min(interval[1], interval[0] + idx_end);
	interval[3] = min(interval[3], interval[2] + (diag + 2) - idx_end);
	interval[0] = interval[0] + idx_start;
	interval[2] = interval[2] + diag - idx_start;
}

void merge_local(const __local float *array,
				 __local float *out,
				 unsigned int interval[4],
				 const unsigned int bit_offset) {
	const unsigned int lid = get_local_id(0);
	unsigned int insertion_index = lid * 2;
//	if (get_global_id(0) == DEBUG_GID)
//		printf("g%d  index: %d\n", DEBUG_GID, insertion_index);

	__local float *out_place = out + insertion_index;
	while (interval[0] < interval[1]) {
		if (interval[2] >= interval[3]) {
			*(out_place++) = array[interval[0]++];
		} else {
			*(out_place++) = get_radix_value(array[interval[0]], bit_offset) < get_radix_value(array[interval[2]], bit_offset) ? array[interval[0]++] : array[interval[2]++];
		}
	}
	while (interval[2] < interval[3]) {
		*(out_place++) = array[interval[2]++];
	}
}

void sort_local(__local float *localIn,
				__local float *localOut,
				const unsigned int left_size,
				const unsigned int right_size,
				const unsigned int sort_size,
				const unsigned int bit_offset) {
	unsigned int interval[4];
	if (get_local_size(0) == sort_size) {
		find_partition_local(localIn, interval, left_size, right_size, sort_size, bit_offset);
	} else {
		find_partition_local(localIn, interval, min(left_size, sort_size), min(right_size, sort_size), sort_size, bit_offset);
	}

	merge_local(localIn, localOut, interval, bit_offset);
	barrier(CLK_LOCAL_MEM_FENCE);
}

void merge_small(__global unsigned int *array,
				 __local unsigned int localArr[2][WORKGROUP_SIZE * 2],
				 int *cur_list,
				 const unsigned int n,
				 const int bit_offset) {
	const unsigned int lid = get_local_id(0);
	const unsigned int workSize = get_local_size(0);
	const unsigned int chunk_num = get_group_id(0);

	*cur_list = 0;

	unsigned int idx = lid + chunk_num * workSize * 2;
	if (idx < n)
		localArr[*cur_list][lid] = array[idx];
	unsigned int size1 = min(workSize, (int)n - chunk_num * workSize * 2);
	int size2 = 0;
	if (size1 + chunk_num * workSize * 2 < n)
	{
		idx += workSize;
		if (idx < n)
			localArr[*cur_list][lid + workSize] = array[idx];
		size2 = min(workSize, n - chunk_num * workSize * 2 - size1);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if (lid + chunk_num * WORKGROUP_SIZE == 0) {
		printf("g!");
		for (int i = 0; i < size1 + size2; ++i) {
			printf("%d ", get_radix_value(localArr[*cur_list][i], bit_offset));
		}
		printf("\n");
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	printf("%d (%d, %d) \n",lid + chunk_num * WORKGROUP_SIZE, size1, size2);

	for (unsigned int sort_size = 1; sort_size < workSize; sort_size <<= 1) {
		sort_local(localArr[*cur_list], localArr[1 - *cur_list], size1, size2, sort_size, bit_offset);
		*cur_list = 1 - *cur_list;
	}

	if (lid + chunk_num * WORKGROUP_SIZE == 0) {
		printf("g!");
		for (int i = 0; i < size1 + size2; ++i) {
			printf("%d ", get_radix_value(localArr[*cur_list][i], bit_offset));
		}
		printf("\n");
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	// input into array
//	idx = lid + chunk_num * workSize * 2;
//	if (idx < n)
//		array[idx] = localArr[cur_list][lid];
//	int max_idx = chunk_num * workSize * 2 + workSize;
//	if (max_idx < n)
//	{
//		idx += workSize;
//		if (idx < n)
//			array[idx] = localArr[cur_list][lid + workSize];
//	}
}

void count_cur_value(__global unsigned int *as,
					 __local unsigned int chunk_arr[2][WORKGROUP_SIZE * 2],
					 const int n,
					 const int bit_offset) {
	const unsigned int lid = get_local_id(0);
	const unsigned int bins_num = 1 << RADIX_BITS;
	const unsigned int chunk = get_group_id(0);
	const unsigned int chunk_size = WORKGROUP_SIZE;
//	printf("%d\n", chunk * chunk_size + 0);
//	printf("%d\n", chunk * chunk_size + 0);
	int cur_bin = lid;
	int count = 0;
	if (cur_bin < bins_num) {
		for (int i = 0; i < chunk_size; ++i) {
//			if (lid == 0) {
//				printf("%d\n", chunk * chunk_size + i);
//				printf("%d\n", chunk * chunk_size + i);
//				printf("%d\n", chunk * chunk_size + i);
//				printf("%d\n", chunk * chunk_size + i);
//			}
			if (chunk * chunk_size + i >= n)
				break;
			if (lid == 0)
				chunk_arr[1][i] = as[chunk * chunk_size + i];
			unsigned int cur_val = get_radix_value(chunk_arr[1][i], bit_offset);
			if (cur_val == cur_bin) {
				chunk_arr[0][i] = count++;
			}
		}
	}
}

__kernel void radix(__global unsigned int *as,
					__global const unsigned int *buffer,
					const int n,
					const int bit_offset) {
	// buffer is `chunks` wide, 'bins` tall
	const unsigned int lid = get_local_id(0);
	const unsigned int bins_num = 1 << RADIX_BITS;
	const unsigned int chunk = get_group_id(0);
	const unsigned int chunk_size = WORKGROUP_SIZE;
	unsigned int chunks_num = (n + WORKGROUP_SIZE - 1) / WORKGROUP_SIZE;

	// sort every chunk locally
	__local unsigned int chunk_arr[2][WORKGROUP_SIZE];
	int cur_list;
	// TODO
//	merge_small(as, chunk_arr, &cur_list, n, bit_offset);
	count_cur_value(as, chunk_arr, n, bit_offset);
	barrier(CLK_LOCAL_MEM_FENCE);
//	printf("%d ", lid + chunk * chunk_size);
	// index of item = prefix for item + (index within chunk - sum of (items of lesser value within chunk))

//	const unsigned int local_item_index = lid;
	const unsigned int local_item_index = chunk_arr[0][lid];
//	const unsigned int item_value = get_radix_value(chunk_arr[1][local_item_index], bit_offset);
	const unsigned int item_value = get_radix_value(chunk_arr[1][lid], bit_offset);

	unsigned int lesser_values_within_chunk = 0;
	for (int i = 0; i < item_value; ++i) {
		lesser_values_within_chunk += buffer[i * chunks_num + chunk] - ((chunk == 0 && i == 0 && lid == 0) ? 0 : buffer[i * chunks_num + chunk - 1]);
	}

	const unsigned int item_index = ((chunk == 0 && item_value == 0 && lid == 0) ? 0 : buffer[item_value * chunks_num + chunk - 1])
									+ local_item_index;
//									- (lesser_values_within_chunk);
	as[item_index] = chunk_arr[cur_list][lid];
}
