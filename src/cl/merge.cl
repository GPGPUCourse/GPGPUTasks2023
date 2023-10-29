#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORKGROUP_SIZE 16

#define DEBUG_GID 33

unsigned int find_local_merge_path(const __local float *array,
							 const unsigned int start0,
							 const unsigned int count0,
							 const unsigned int start1,
							 const unsigned int count1,
							 const unsigned int diag) {
	unsigned int l = max(0, (int)diag - (int)count1);
	unsigned int r = min(diag, count0);
	unsigned int m;
	while (l < r) {
		m = (l + r) >> 1;
		if (array[start0 + m] < array[start1 + diag - 1 - m]) {
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
						  const unsigned int sortedSize) {
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
										  diag);
	int idx_end = find_local_merge_path(array,
										  interval[0],
										  interval[1] - interval[0],
										  interval[2],
										  interval[3] - interval[2],
										  diag + 2);
	interval[1] = min(interval[1], interval[0] + idx_end);
	interval[3] = min(interval[3], interval[2] + (diag + 2) - idx_end);
	interval[0] = interval[0] + idx_start;
	interval[2] = interval[2] + diag - idx_start;
}

void merge_local(const __local float *array,
				 __local float *out,
				 unsigned int interval[4]) {
	const unsigned int lid = get_local_id(0);
	unsigned int insertion_index = lid * 2;
//	if (get_global_id(0) == DEBUG_GID)
//		printf("g%d  index: %d\n", DEBUG_GID, insertion_index);

	__local float *out_place = out + insertion_index;
	while (interval[0] < interval[1]) {
		if (interval[2] >= interval[3]) {
			*(out_place++) = array[interval[0]++];
		} else {
			*(out_place++) = array[interval[0]] < array[interval[2]] ? array[interval[0]++] : array[interval[2]++];
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
				const unsigned int sort_size) {
	unsigned int interval[4];
	if (get_local_size(0) == sort_size) {
		find_partition_local(localIn, interval, left_size, right_size, sort_size);
	} else {
		find_partition_local(localIn, interval, min(left_size, sort_size), min(right_size, sort_size), sort_size);
	}

	if (get_global_id(0) == DEBUG_GID) {
		printf("g (%d, %d)\n", left_size, right_size);
	}

	if (get_global_id(0) == DEBUG_GID) {
		printf("g aftr [%d, %d]-[%d, %d]\n", interval[0], interval[1], interval[2], interval[3]);
	}
	merge_local(localIn, localOut, interval);
	barrier(CLK_LOCAL_MEM_FENCE);
}

//unsigned int min(unsigned int x, unsigned int y) {
//	return x > y ? x : y;
//}

__kernel void merge_small(__global float *array,
						  const unsigned int n) {
	const unsigned int lid = get_local_id(0);
	const unsigned int workSize = get_local_size(0);
	const unsigned int chunk_num = get_group_id(0);
	__local float localArr[2][WORKGROUP_SIZE * 2];

	int cur_list = 0;

	unsigned int idx = lid + chunk_num * workSize * 2;
	if (idx < n)
		localArr[cur_list][lid] = array[idx];
	unsigned int size1 = min(workSize, (int)n - chunk_num * workSize * 2);
	int size2 = 0;
	if (size1 + chunk_num * workSize * 2 < n)
	{
		idx += workSize;
		if (idx < n)
			localArr[cur_list][lid + workSize] = array[idx];
		size2 = min(workSize, n - chunk_num * workSize * 2 - size1);
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	for (unsigned int sort_size = 1; sort_size <= workSize; sort_size <<= 1) {
		sort_local(localArr[cur_list], localArr[1 - cur_list], size1, size2, sort_size);
		cur_list = 1 - cur_list;
	}

	// input into array
	idx = lid + chunk_num * workSize * 2;
	if (idx < n)
		array[idx] = localArr[cur_list][lid];
	int max_idx = chunk_num * workSize * 2 + workSize;
	if (max_idx < n)
	{
		idx += workSize;
		if (idx < n)
			array[idx] = localArr[cur_list][lid + workSize];
	}
}

// find first item in interval that is less than or equal to *element*.
unsigned int find_merge_path(const __global float *array,
					 const unsigned int start0,
					 const unsigned int count0,
					 const unsigned int start1,
					 const unsigned int count1,
					 const unsigned int diag) {
	unsigned int l = max(0, (int)diag - (int)count1);
	unsigned int r = min(diag, count0);
	unsigned int m;
	while (l < r) {
		m = (l + r) >> 1;
		if (array[start0 + m] < array[start1 + diag - 1 - m]) {
			l = m + 1;
		} else {
			r = m;
		}
	}
	return l;
}

// finds borders of two arrays of size <= workGroupSize to sort
__kernel void find_partitions(const __global float *array,
							 __global unsigned int *intervals,
							 const unsigned int n,
							 const unsigned int sortedSize) {
	// intervals: all starts[0], all ends[0], all starts[1], all ends[1].
	const unsigned int gid = get_global_id(0);
	const unsigned int workGroups = get_global_size(0);
	const unsigned int partitions = sortedSize / get_local_size(0);
	const unsigned int global_offset = min(n, (gid / partitions) * sortedSize * 2);
	const unsigned int workGroupSize = get_local_size(0);

	unsigned int pid = gid % partitions;
	unsigned int l, r, m;
	unsigned int diag1 = pid * (workGroupSize * 2);
	unsigned int diag2 = (pid + 1) * (workGroupSize * 2);

//	const unsigned int lid = get_local_id(0);
//	interval[0] = (lid / sortedSize) * sortedSize * 2;
//	interval[1] = interval[0] + leftSize;
//	interval[2] = interval[1];
//	interval[3] = interval[2] + rightSize;
//
//	unsigned int diag = (lid % sortedSize) * 2;


	unsigned int interval[4];

	interval[0] = global_offset;
	interval[1] = interval[0] + sortedSize;
	interval[2] = interval[1];
	interval[3] = interval[2] + sortedSize;

//	if (get_global_id(0) == DEBUG_GID)
//		printf("g%d before %d [%u, %u]-[%u, %u]; %d, %d\n", DEBUG_GID, sortedSize, interval[0], interval[1], interval[2], interval[3], diag1, diag2);

	int idx_start = find_merge_path(array,
										  interval[0],
										  interval[1] - interval[0],
										  interval[2],
										  interval[3] - interval[2],
										  diag1);
	int idx_end = find_merge_path(array,
										interval[0],
										interval[1] - interval[0],
										interval[2],
										interval[3] - interval[2],
										diag2);
	interval[1] = min(interval[1], interval[0] + idx_end);
	interval[3] = min(interval[3], interval[2] + diag2 - idx_end);
	interval[0] = interval[0] + idx_start;
	interval[2] = interval[2] + diag1 - idx_start;

	intervals[gid] = interval[0];
	intervals[gid + workGroups] = interval[1];
	intervals[gid + workGroups * 2] = interval[2];
	intervals[gid + workGroups * 3] = interval[3];
}

__kernel void merge(const __global float *array,
					__global float *buffer,
					const __global unsigned int *intervals,
					const unsigned int n) {
	const unsigned int gid = get_global_id(0);
	const unsigned int group_id = get_group_id(0);
	const unsigned int workGroups = get_global_size(0);

	const unsigned int lid = get_local_id(0);
	const unsigned int workSize = get_local_size(0);
	const int a = n / workSize;
	if (workSize > a)
		return;

	unsigned int starts[2] = {intervals[group_id], intervals[group_id + workGroups * 2]};
	unsigned int ends[2] = {intervals[group_id + workGroups * 1], intervals[group_id + workGroups * 3]};


	__local float localArr[2][WORKGROUP_SIZE * 2];

	int cur_list = 0;

	unsigned int idx = starts[0] + lid;
	unsigned int l_idx = lid;
//	if (get_global_id(0) == DEBUG_GID) {
//		printf("g idx0: %d\n", idx);
//	}
	while (idx < ends[0]) {
//		if (get_group_id(0) == 2)
//			printf("%d ", idx);
		localArr[cur_list][l_idx] = array[idx];
		l_idx += workSize;
		idx += workSize;
//		if (get_global_id(0) == DEBUG_GID) {
//			printf("g idx0: %d\n", idx);
//		}
	}
//	if (get_global_id(0) == DEBUG_GID) {
//		printf("\n");
//	}

	barrier(CLK_LOCAL_MEM_FENCE);
//	if (get_global_id(0) == DEBUG_GID) {
//		printf("g 1 [%d, %d] - [%d, %d]\n", starts[0], ends[0], starts[1], ends[1]);
//		for (int i = 0; i < ends[0] - starts[0] + ends[1] - starts[1]; ++i) {
//			printf("%f ", localArr[0][i]);
//		}
//		printf("\n");
//	}
	barrier(CLK_LOCAL_MEM_FENCE);
	idx = starts[1] + lid;
	l_idx = lid;
//	if (get_global_id(0) == DEBUG_GID) {
//		printf("g idx1: %d; array[idx]=%f\n", idx, array[idx]);
//	}
	while (idx < ends[1]) {
		localArr[cur_list][lid + (ends[0] - starts[0])] = array[idx];
		idx += workSize;
		l_idx += workSize;
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if (get_global_id(0) == DEBUG_GID) {
		printf("g 2 [%d, %d] - [%d, %d]\n", starts[0], ends[0], starts[1], ends[1]);
		for (int i = 0; i < ends[0] - starts[0] + ends[1] - starts[1]; ++i) {
			printf("%f ", localArr[0][i]);
		}
		printf("\n");
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	sort_local(localArr[cur_list], localArr[1 - cur_list], ends[0] - starts[0], ends[1] - starts[1], workSize);
	cur_list = 1 - cur_list;
//	if (get_global_id(0) == DEBUG_GID) {
//		printf("g [%d, %d] - [%d, %d]\n", starts[0], ends[0], starts[1], ends[1]);
//		for (int i = 0; i < ends[0] - starts[0] + ends[1] - starts[1]; ++i) {
//			printf("%f ", localArr[cur_list][i]);
//		}
//		printf("\n");
//	}

	// input into array
	idx = lid + group_id * workSize * 2;
//	if (get_global_id(0) == DEBUG_GID) {
//		printf("g%d to: %d\n", DEBUG_GID, idx);
//	}
	if (idx < n)
		buffer[idx] = localArr[cur_list][lid];
	int max_idx = group_id * workSize * 2 + workSize;
	if (max_idx < n)
	{
		idx += workSize;
		if (idx < n)
			buffer[idx] = localArr[cur_list][lid + workSize];
//		if (get_global_id(0) == DEBUG_GID) {
//			printf("g%d to1: %d\n", DEBUG_GID, idx);
//		}
	}

//
//	__global float *buf_place = buffer + insertion_index;
//	while (starts[0] < ends[0]) {
//		if (starts[1] >= ends[1]) {
//			*(buf_place++) = array[starts[0]++];
//		} else {
//			*(buf_place++) = array[starts[0]] < array[starts[1]] ? array[starts[0]++] : array[starts[1]++];
//		}
//	}
//	while (starts[1] < ends[1]) {
//		*(buf_place++) = array[starts[1]++];
//	}

}
//
//__kernel void merge_sort(const __global float *array,
//						 __global float *buffer,
//						 __global const unsigned int *intervals,
//						 const unsigned int n,
//						 const unsigned int sortedSize) {
//	// intervals: all starts[0], all ends[0], all starts[1], all ends[1].
//
////	unsigned int workgroup_size = get_local_size(0);
////	if (sortedSize < workgroup_size) {
////		sort_local(array, n);
////		return;
////	}
//
//	const unsigned int gid = get_global_id(0);
//	const unsigned int partitions = sortedSize * 2 / WORK_PER_THREAD;
//	unsigned int starts[2] = {min(n, (gid / partitions) * sortedSize * 2), min(n, (gid / partitions) * sortedSize * 2 + sortedSize)};
//	unsigned int ends[2] = {min(n, (gid / partitions) * sortedSize * 2 + sortedSize), min(n, (gid / partitions + 1) * sortedSize * 2)};
//
//	find_partition_interval(array, starts, ends, sortedSize);
//	merge(array, buffer, starts, ends, gid * WORK_PER_THREAD);
//}