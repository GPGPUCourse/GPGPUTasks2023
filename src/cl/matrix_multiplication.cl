#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void matrix_multiplication_global_mem(
                    __global const float* as, 
                    __global const float* bs, 
                    __global float* cs,
                    const unsigned int M, 
                    const unsigned int K,
                    const unsigned int N)
{
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);
    
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += as[y * K + k] * bs[k * N + x];
    }
    cs[y * N + x] = sum;
}

__kernel void matrix_multiplication_local_mem(
                    __global const float* as, 
                    __global const float* bs, 
                    __global float* cs,
                    const unsigned int M, 
                    const unsigned int K,
                    const unsigned int N)
{
    const unsigned x = get_global_id(0);
    const unsigned y = get_global_id(1);

    const unsigned n = get_local_size(0); // WG0
    const unsigned m = get_local_size(1); // WG1 
    const unsigned k = D;
    const unsigned i = get_local_id(0);
    const unsigned j = get_local_id(1);

    const int gi = get_group_id(0);
    const int gj = get_group_id(1);

    __local float suba[WG1][D]; 
    __local float subb[D][WG0]; 

    float sum = 0.0;
    for (int s = 0; s * k < K; ++s) {
        if(i<k) {
            suba[j][i] = as[gj*K*m+j*K+s*k+i];
        }
        if(j<k) {
            subb[j][i] = bs[s*N*k+j*N+gi*n+i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int z = 0; z < k; ++z) {
            sum += suba[j][z] * subb[z][i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    cs[y * N + x] = sum;
}

__kernel void matrix_multiplication_local_mem2(
                    __global const float* as, 
                    __global const float* bs, 
                    __global float* cs,
                    const unsigned int M, 
                    const unsigned int K,
                    const unsigned int N)
{
    const unsigned n = get_local_size(0); // WG0
    const unsigned m = D;
    const unsigned i = get_local_id(0);
    const unsigned j = get_local_id(1);

    const int gi = get_group_id(0);
    const int gj = get_group_id(1);

    float sum[D];
    for (int d = 0; d < m; ++d)
        sum[d] = 0.0f;

    __local float suba[D][D]; 
    __local float subb[D][WG0]; 
    for(int s = 0; s * m < K; ++s) {
        for(int d = 0; d < m; ++d) {
            if(i < m) {
                suba[d][i] = as[gj*m*K+d*K+s*m+i];
            }
            subb[d][i] = bs[s*N*m+d*N+gi*n+i];
        }

        barrier(CLK_LOCAL_MEM_FENCE);

        for (int d = 0; d < m; ++d) {
            for (int t = 0; t < m; ++t) {
                sum[d] += suba[d][t] * subb[t][i];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int d = 0; d < m; ++d)
        cs[gj*m*N+d*N+gi*n+i] = sum[d];
}