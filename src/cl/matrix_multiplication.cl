#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 6

#define WORKGROUP_SIZE 16

//__kernel void matrix_transpose(__global const float *input,
//							   __global float *output,
//							   const int M,
//							   const int K)
//{
//	const unsigned int lid_x = get_local_id(0);
//	const unsigned int lid_y = get_local_id(1);
//	const unsigned int gid_x = get_global_id(0);
//	const unsigned int gid_y = get_global_id(1);
//	const unsigned int wid_x = get_group_id(0);
//	const unsigned int wid_y = get_group_id(1);
//
//	__local float buffer[WORKGROUP_SIZE * WORKGROUP_SIZE];
//	buffer[lid_y * WORKGROUP_SIZE + (lid_x + lid_y) % WORKGROUP_SIZE] = input[gid_y * M + gid_x];
//
//	barrier(CLK_LOCAL_MEM_FENCE);
//
//	output[index_y * K + index_x] = buffer[lid_x * WORKGROUP_SIZE + (lid_y + lid_x) % WORKGROUP_SIZE];
//}

__kernel void matrix_multiplication_1(__global const float *input_1,
									  __global const float *input_2,
									  __global float *output,
									  const int M,
									  const int K,
									  const int N)
{
	const unsigned int gid_x = get_global_id(0);
	const unsigned int gid_y = get_global_id(1);

	int sum = 0;
	for (int i = 0; i < K; ++i) {
		sum += input_1[gid_y * K + i] * input_2[i * N + gid_x];
	}

	output[gid_y * M + gid_x] = sum;
}



__kernel void matrix_multiplication_2(__global const float *input_1,
									  __global const float *input_2,
									  __global float *output,
									  const int M,
									  const int K,
									  const int N)
{
	const unsigned int gid_x = get_global_id(0);
	const unsigned int gid_y = get_global_id(1);
	const unsigned int lid_x = get_local_id(0);
	const unsigned int lid_y = get_local_id(1);

	__local float tileA[WORKGROUP_SIZE][WORKGROUP_SIZE];
	__local float tileB[WORKGROUP_SIZE][WORKGROUP_SIZE];

	int sum = 0;
	for (int workgroup_offset = 0; workgroup_offset < K; workgroup_offset += WORKGROUP_SIZE) {
		tileA[lid_y][lid_x] = input_1[gid_y * K + (workgroup_offset + lid_x)];
		tileB[lid_y][(lid_x + lid_y) % WORKGROUP_SIZE] = input_2[(workgroup_offset + lid_y) * N + gid_x];
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int k = 0; k < WORKGROUP_SIZE; ++k){
			sum += tileA[lid_y][k] * tileB[k][(lid_x + k) % WORKGROUP_SIZE];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	output[gid_y * N + gid_x] = sum;
}