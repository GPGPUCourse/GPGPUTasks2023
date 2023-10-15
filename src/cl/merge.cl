#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORK_PER_THREAD 2

void sort_single(__global float *array,
						  const unsigned int n,
						  const unsigned int sortedSize) {
	const unsigned int gid = get_global_id(0);
	float list1[WORK_PER_THREAD];
	float list2[WORK_PER_THREAD];
	int size1 = 0;
	int size2 = 0;

	for (int i = 0; i < sortedSize; ++i, ++size1) {
		const unsigned int idx = gid * sortedSize * 2 + i;
		if (idx >= n)
			break;
		list1[i] = array[idx];
	}
	for (int i = 0; i < sortedSize; ++i, ++size2) {
		const unsigned int idx = gid * sortedSize * 2 + sortedSize + i;
		if (idx >= n)
			break;
		list2[i] = array[idx];
	}

	int i = 0;
	int j = 0;
	__global float *arr_place = array + gid * sortedSize * 2;
	while (i < size1) {
		if (j >= size2) {
			*(arr_place++) = list1[i++];
		} else {
			*(arr_place++) = list1[i] < list2[j] ? list1[i++] : list2[j++];
		}
	}
	while (j < size2) {
		*(arr_place++) = list2[j++];
	}
}

// find first item in interval that is less than or equal to *element*.
unsigned int binary_search(const __global float *array,
						   unsigned int start,
						   unsigned int end,
						   float element) {
	// end is NOT included.
	while (end - start > 1) {
		unsigned int mid = (start + end) / 2;
		if (array[mid] > element) {
			end = mid;
		} else {
			start = mid;
		}
	}
	return start;
}

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


void find_partition_interval(const __global float *array,
							 unsigned int *starts,
							 unsigned int *ends,
							 unsigned int sortedSize) {
	const unsigned int gid = get_global_id(0);
	const unsigned int partitions = sortedSize * 2 / WORK_PER_THREAD;
	unsigned int pid = gid % partitions;
	unsigned int l, r, m;
	unsigned int diag1 = pid * WORK_PER_THREAD;
	unsigned int diag2 = (pid + 1) * WORK_PER_THREAD;

	int idx_start = 0;
	int idx_end = sortedSize;
	if (pid != 0) {
		idx_start = find_merge_path(array, starts[0], ends[0] - starts[0], starts[1], ends[1] - starts[1], diag1);
	}
	if (pid != partitions - 1) {
		idx_end = find_merge_path(array, starts[0], ends[0] - starts[0], starts[1], ends[1] - starts[1], diag2);
	}
	ends[0] = min(ends[0], starts[0] + idx_end);
	ends[1] = min(ends[1], starts[1] + diag2 - idx_end);
	starts[0] = starts[0] + idx_start;
	starts[1] = starts[1] + diag1 - idx_start;
}

void merge(const __global float *array,
		   __global float *buffer,
		   unsigned int *starts,
		   unsigned int *ends,
		   unsigned int insertion_index) {
	__global float *buf_place = buffer + insertion_index;
	while (starts[0] < ends[0]) {
		if (starts[1] >= ends[1]) {
			*(buf_place++) = array[starts[0]++];
		} else {
			*(buf_place++) = array[starts[0]] < array[starts[1]] ? array[starts[0]++] : array[starts[1]++];
		}
	}
	while (starts[1] < ends[1]) {
		*(buf_place++) = array[starts[1]++];
	}

}

__kernel void merge_sort(__global float *array,
						 __global float *buffer,
						 const unsigned int n,
						 const unsigned int sortedSize) {
	if (sortedSize <= WORK_PER_THREAD) {
		sort_single(array, n, sortedSize);
		return;
	}

	const unsigned int gid = get_global_id(0);
	const unsigned int partitions = sortedSize * 2 / WORK_PER_THREAD;
	unsigned int starts[2] = {min(n, (gid / partitions) * sortedSize * 2), min(n, (gid / partitions) * sortedSize * 2 + sortedSize)};
	unsigned int ends[2] = {min(n, (gid / partitions) * sortedSize * 2 + sortedSize), min(n, (gid / partitions + 1) * sortedSize * 2)};

	find_partition_interval(array, starts, ends, sortedSize);
	merge(array, buffer, starts, ends, gid * WORK_PER_THREAD);
}