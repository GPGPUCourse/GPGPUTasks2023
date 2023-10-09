#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define SINGLE_ITEM_WORK 256

void sort_single(__global float *array,
						  const unsigned int n,
						  const unsigned int sortedSize) {
	const unsigned int gid = get_global_id(0);
	float list1[sortedSize];
	float list2[sortedSize];
	int size1 = 0;
	int size2 = 0;

	for (int i = 0; i < sortedSize; ++i, ++size1) {
		const unsigned int idx = gid * sortedSize * 2 + i;
		if (idx >= n)
			break;
		list1[i] = array[idx];
	}
	for (int i = 0; i < sortedSize; ++i, ++size2) {
		const unsigned int idx = gid * sortedSize + sortedSize + i;
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

__kernel void merge_sort(__global float *array,
					__global float *buffer,
					const unsigned int n,
					const unsigned int sortedSize) {
	if (sortedSize <= SINGLE_ITEM_WORK) {
		sort_single(array, n, sortedSize);
	} else {
		// TODO
		printf("I don't know what to do past this point!!! ;(\n");
	}
}