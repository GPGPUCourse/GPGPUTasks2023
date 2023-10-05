#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORKGROUP_SIZE 128
#define MAX_WORK_SINGLE 256

// Small blocks sorting
// Sorts as[l..r) in local memory
// 0 <= r - l <= MAX_WORK_SINGLE should hold
__kernel void merge_not_parallel(__global float * as,
                                 const unsigned int l,
                                 const unsigned int r)
{
    float merge_mem1[MAX_WORK_SINGLE], merge_mem2[MAX_WORK_SINGLE];
    float *from = merge_mem1, *to = merge_mem2;
    const unsigned int tot_len = r - l;
    for (int i = 0; i < tot_len; i++)
    {
        merge_mem1[i] = as[i + l];
    }
    for (int len = 1; len < tot_len; len <<= 1)
    {
        int offset = 0;
        for (; offset + len < tot_len; offset += 2 * len)
        {
            int pnt1 = 0;
            int pnt2 = 0;
            while (pnt1 < len && pnt2 < len && offset + len + pnt2 < tot_len)
            {
                if (from[offset + pnt1] < from[offset + len + pnt2])
                {
                    to[offset + pnt1 + pnt2] = from[offset + pnt1];
                    pnt1++;
                }
                else
                {
                    to[offset + pnt1 + pnt2] = from[offset + len + pnt2];
                    pnt2++;
                }
            }
            while (pnt1 < len)
            {
                to[offset + pnt1 + pnt2] = from[offset + pnt1];
                pnt1++;
            }
            while (pnt2 < len && offset + len + pnt2 < tot_len)
            {
                to[offset + pnt1 + pnt2] = from[offset + len + pnt2];
                pnt2++;
            }
        }

        while (offset < tot_len)
        {
            to[offset] = from[offset];
            offset++;
        }

        float *tmp = to;
        to = from;
        from = tmp;
    }
    for (int i = 0; i < tot_len; i++)
    {
        as[i + l] = from[i];
    }
}

__kernel void merge_one_workgroup(__global float * as,
                                  __global float * bs,
                                  const unsigned int n)
{
    unsigned int id = get_local_id(0);

    for (unsigned int offset = 0; offset < n; offset += MAX_WORK_SINGLE * WORKGROUP_SIZE)
    {
        unsigned int l = offset + id * MAX_WORK_SINGLE;
        unsigned int r = offset + (id + 1) * MAX_WORK_SINGLE;

        if (r <= n)
            merge_not_parallel(as, l, r);
        else if (l < n)
            merge_not_parallel(as, l, n);
    }

    barrier(CLK_GLOBAL_MEM_FENCE);

    __local unsigned int ls[WORKGROUP_SIZE + 1], rs[WORKGROUP_SIZE + 1];
    for (unsigned int len = MAX_WORK_SINGLE; len < n; len <<= 1)
    {
        unsigned int itemsPerWorkflow = 2 * len / WORKGROUP_SIZE;
        unsigned int offset = 0;

        for (; offset + len < n; offset += 2 * len)
        {
            unsigned int pre = id * itemsPerWorkflow;
            if (offset + pre < n)
            {
                int l = pre > len ? pre - len : 0;
                if (l + (n - offset - len) < pre) l = pre - (n - offset - len);
                l--;

                int r = min(pre, len);
                while (l + 1 < r)
                {
                    int m = (l + r) / 2;
                    if (as[offset + m] < as[offset + len + pre - 1 - m])
                        l = m;
                    else
                        r = m;
                }
                ls[id] = offset + r, rs[id] = offset + len + pre - r;
            }
            else
            {
                ls[id] = offset + len, rs[id] = n;
            }

            if (id == 0)
            {
                ls[WORKGROUP_SIZE] = offset + len, rs[WORKGROUP_SIZE] = min(offset + 2 * len, n);
            }

            barrier(CLK_LOCAL_MEM_FENCE);

            int pnt1 = 0, pnt2 = 0;
            while (ls[id] + pnt1 < ls[id + 1] && rs[id] + pnt2 < rs[id + 1])
            {
                if (as[ls[id] + pnt1] < as[rs[id] + pnt2])
                {
                    bs[offset + pre + pnt1 + pnt2] = as[ls[id] + pnt1];
                    pnt1++;
                }
                else
                {
                    bs[offset + pre + pnt1 + pnt2] = as[rs[id] + pnt2];
                    pnt2++;
                }
            }
            while (ls[id] + pnt1 < ls[id + 1])
            {
                bs[offset + pre + pnt1 + pnt2] = as[ls[id] + pnt1];
                pnt1++;
            }
            while (rs[id] + pnt2 < rs[id + 1])
            {
                bs[offset + pre + pnt1 + pnt2] = as[rs[id] + pnt2];
                pnt2++;
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }

        if (id == 0)
        {
            while (offset < n)
            {
                bs[offset] = as[offset];
                offset++;
            }
        }

        __global float *tmp = bs;
        bs = as;
        as = tmp;

        barrier(CLK_GLOBAL_MEM_FENCE);
    }

    for (int i = 0; i < n; i += WORKGROUP_SIZE)
    {
        if (i + id < n)
        {
            bs[i + id] = as[i + id];
        }
    }
}