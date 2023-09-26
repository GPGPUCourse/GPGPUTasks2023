__kernel void matrix_multiplication(...)
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
		sum += input_1[gid_y * M + i] * input_2[i * K + gid_x];
	}

	output[gid_y * M + gid_x] = sum;
}