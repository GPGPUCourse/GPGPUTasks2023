#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GROUP_SIZE 16
#define THREAD_WORK 8
#define matrix(as, i, j, n, m) as[(i) * (m) + (j)]
#define matrixSafeGet(as, i, j, n, m) ((i) < (n) && (j) < (m) ? as[(i) * (m) + (j)] : 0.0f)

__kernel void matrix_multiplication_more_work_per_thread3(__global const float * as,
                                                          __global const float * bs,
                                                          __global float * cs,
                                                          const unsigned int m,
                                                          const unsigned int k,
                                                          const unsigned int n)
{
    unsigned int gi = get_group_id(1);
    unsigned int gj = get_group_id(0);
    unsigned int li = get_local_id(1);
    unsigned int lj = get_local_id(0);

    __local float as_local[GROUP_SIZE][GROUP_SIZE];
    __local float bs_local[GROUP_SIZE][THREAD_WORK][GROUP_SIZE];

    float sum[THREAD_WORK];
    for (int t = 0; t < THREAD_WORK; t++) sum[t] = 0.0f;

    for (int t = 0; t < k; t += GROUP_SIZE)
    {
        as_local[lj][li] = matrixSafeGet(as, gi * GROUP_SIZE + li, t + lj, m, k);
        for (int s = 0; s < THREAD_WORK; s++)
        {
            bs_local[li][s][lj] = matrixSafeGet(bs, t + li, gj * GROUP_SIZE * THREAD_WORK + lj * THREAD_WORK + s, k, n);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int tt = 0; tt < GROUP_SIZE; tt++)
        {
            float tmp = as_local[tt][li];
            for (int s = 0; s < THREAD_WORK; s++)
            {
                sum[s] += tmp * bs_local[tt][s][lj];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int s = 0; s < THREAD_WORK; s++)
    {
        if (gi * GROUP_SIZE + li < m && gj * GROUP_SIZE * THREAD_WORK + lj * THREAD_WORK + s < n)
        {
            matrix(cs, gi * GROUP_SIZE + li, gj * GROUP_SIZE * THREAD_WORK + lj * THREAD_WORK + s, m, n) = sum[s];
        }
    }
}

__kernel void matrix_multiplication_more_work_per_thread2(__global const float * as,
                                                         __global const float * bs,
                                                         __global float * cs,
                                                         const unsigned int m,
                                                         const unsigned int k,
                                                         const unsigned int n)
{
    unsigned int gi = get_group_id(1);
    unsigned int gj = get_group_id(0);
    unsigned int li = get_local_id(1);
    unsigned int lj = get_local_id(0);

    __local float as_local[GROUP_SIZE][GROUP_SIZE];
    __local float bs_local[THREAD_WORK][GROUP_SIZE][GROUP_SIZE];

    float sum[THREAD_WORK];
    for (int t = 0; t < THREAD_WORK; t++) sum[t] = 0.0f;

    for (int t = 0; t < k; t += GROUP_SIZE)
    {
        as_local[li][lj] = matrixSafeGet(as, gi * GROUP_SIZE + li, t + lj, m, k);
        for (int s = 0; s < THREAD_WORK; s++)
        {
            bs_local[s][li][lj] = matrixSafeGet(bs, t + li, gj * GROUP_SIZE * THREAD_WORK + lj * THREAD_WORK + s, k, n);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int s = 0; s < THREAD_WORK; s++)
        {
            for (int tt = 0; tt < GROUP_SIZE; tt++)
            {
                sum[s] += as_local[li][tt] * bs_local[s][tt][lj];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int s = 0; s < THREAD_WORK; s++)
    {
        if (gi * GROUP_SIZE + li < m && gj * GROUP_SIZE * THREAD_WORK + lj * THREAD_WORK + s < n)
        {
            matrix(cs, gi * GROUP_SIZE + li, gj * GROUP_SIZE * THREAD_WORK + lj * THREAD_WORK + s, m, n) = sum[s];
        }
    }
}

__kernel void matrix_multiplication_more_work_per_thread(__global const float * as,
                                                         __global const float * bs,
                                                         __global float * cs,
                                                         const unsigned int m,
                                                         const unsigned int k,
                                                         const unsigned int n)
{
    unsigned int gi = get_group_id(1);
    unsigned int gj = get_group_id(0);
    unsigned int li = get_local_id(1);
    unsigned int lj = get_local_id(0);

    __local float as_local[GROUP_SIZE][GROUP_SIZE];
    __local float bs_local[GROUP_SIZE][GROUP_SIZE * THREAD_WORK];

    float sum[THREAD_WORK];
    for (int t = 0; t < THREAD_WORK; t++) sum[t] = 0.0f;

    for (int t = 0; t < k; t += GROUP_SIZE)
    {
        as_local[li][lj] = matrixSafeGet(as, gi * GROUP_SIZE + li, t + lj, m, k);
        for (int s = 0; s < THREAD_WORK; s++)
        {
            bs_local[li][lj * THREAD_WORK + s] = matrixSafeGet(bs, t + li, gj * GROUP_SIZE * THREAD_WORK + lj * THREAD_WORK + s, k, n);
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int tt = 0; tt < GROUP_SIZE; tt++)
        {
            float tmp = as_local[li][tt];
            for (int s = 0; s < THREAD_WORK; s++)
            {
                sum[s] += tmp * bs_local[tt][lj * THREAD_WORK + s];
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (int s = 0; s < THREAD_WORK; s++)
    {
        if (gi * GROUP_SIZE + li < m && gj * GROUP_SIZE * THREAD_WORK + lj * THREAD_WORK + s < n)
        {
            matrix(cs, gi * GROUP_SIZE + li, gj * GROUP_SIZE * THREAD_WORK + lj * THREAD_WORK + s, m, n) = sum[s];
        }
    }
}

__kernel void matrix_multiplication_local_mem_coalesced2(__global const float * as,
                                                        __global const float * bs,
                                                        __global float * cs,
                                                        const unsigned int m,
                                                        const unsigned int k,
                                                        const unsigned int n)
{
    unsigned int gi = get_group_id(1);
    unsigned int gj = get_group_id(0);
    unsigned int li = get_local_id(1);
    unsigned int lj = get_local_id(0);

    __local float as_local[GROUP_SIZE][GROUP_SIZE];
    __local float bs_local[GROUP_SIZE][GROUP_SIZE];
    float sum = 0;
    for (int t = 0; t < k; t += GROUP_SIZE)
    {
        as_local[lj][li] = matrixSafeGet(as, gi * GROUP_SIZE + li, t + lj, m, k);
        bs_local[li][lj] = matrixSafeGet(bs, t + li, gj * GROUP_SIZE + lj, k, n);
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int tt = 0; tt < GROUP_SIZE; tt++)
        {
            sum += as_local[tt][li] * bs_local[tt][lj];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gi * GROUP_SIZE + li >= m || gj * GROUP_SIZE + lj >= n) return;
    matrix(cs, gi * GROUP_SIZE + li, gj * GROUP_SIZE + lj, m, n) = sum;
}

__kernel void matrix_multiplication_local_mem_coalesced(__global const float * as,
                                                        __global const float * bs,
                                                        __global float * cs,
                                                        const unsigned int m,
                                                        const unsigned int k,
                                                        const unsigned int n)
{
    unsigned int gi = get_group_id(1);
    unsigned int gj = get_group_id(0);
    unsigned int li = get_local_id(1);
    unsigned int lj = get_local_id(0);

    __local float as_local[GROUP_SIZE][GROUP_SIZE];
    __local float bs_local[GROUP_SIZE][GROUP_SIZE];
    float sum = 0;
    for (int t = 0; t < k; t += GROUP_SIZE)
    {
        as_local[li][lj] = matrixSafeGet(as, gi * GROUP_SIZE + li, t + lj, m, k);
        bs_local[lj][li] = matrixSafeGet(bs, t + li, gj * GROUP_SIZE + lj, k, n);
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int tt = 0; tt < GROUP_SIZE; tt++)
        {
            sum += as_local[li][tt] * bs_local[lj][tt];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gi * GROUP_SIZE + li >= m || gj * GROUP_SIZE + lj >= n) return;
    matrix(cs, gi * GROUP_SIZE + li, gj * GROUP_SIZE + lj, m, n) = sum;
}

__kernel void matrix_multiplication_local_mem_not_coalesced(__global const float * as,
                                                            __global const float * bs,
                                                            __global float * cs,
                                                            const unsigned int m,
                                                            const unsigned int k,
                                                            const unsigned int n)
{
    unsigned int gi = get_group_id(0);
    unsigned int gj = get_group_id(1);
    unsigned int li = get_local_id(0);
    unsigned int lj = get_local_id(1);

    __local float as_local[GROUP_SIZE][GROUP_SIZE];
    __local float bs_local[GROUP_SIZE][GROUP_SIZE];
    float sum = 0;
    for (int t = 0; t < k; t += GROUP_SIZE)
    {
        as_local[li][lj] = matrixSafeGet(as, gi * GROUP_SIZE + li, t + lj, m, k);
        bs_local[li][lj] = matrixSafeGet(bs, t + li, gj * GROUP_SIZE + lj, k, n);
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int tt = 0; tt < GROUP_SIZE; tt++)
        {
            sum += as_local[li][tt] * bs_local[tt][lj];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (gi * GROUP_SIZE + li >= m || gj * GROUP_SIZE + lj >= n) return;
    matrix(cs, gi * GROUP_SIZE + li, gj * GROUP_SIZE + lj, m, n) = sum;
}

__kernel void matrix_multiplication_naive(__global const float * as,
                                          __global const float * bs,
                                          __global float * cs,
                                          const unsigned int m,
                                          const unsigned int k,
                                          const unsigned int n)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= m || j >= n) return;

    float sum = 0;
    for (unsigned int t = 0; t < k; t++)
    {
        sum += matrix(as, i, t, m, k) * matrix(bs, t, j, k, n);
    }
    matrix(cs, i, j, m, n) = sum;
}

__kernel void matrix_multiplication_naive0(__global const float * as,
                                           __global const float * bs,
                                           __global float * cs,
                                           const unsigned int m,
                                           const unsigned int k,
                                           const unsigned int n)
{
    unsigned int i = get_global_id(0);
    unsigned int j = get_global_id(1);

    if (i >= m || j >= n) return;

    matrix(cs, i, j, m, n) = 0;
    for (unsigned int t = 0; t < k; t++)
    {
        matrix(cs, i, j, m, n) += matrix(as, i, t, m, k) * matrix(bs, t, j, k, n);
    }
}