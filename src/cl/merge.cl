#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORKGROUP_SIZE 128
#define MAX_WORK_SINGLE 256

__kernel void merge_not_parallel(__global float * as,
                                 const unsigned int l,
                                 const unsigned int r)
{
    float merge_mem1[MAX_WORK_SINGLE], merge_mem2[MAX_WORK_SINGLE];
    const unsigned int tot_len = r - l;
    for (int i = 0; i < tot_len; i++)
    {
        merge_mem1[i] = as[i + l];
    }
    int direction = 0;
    for (int len = 1; len < tot_len; len <<= 1, direction ^= 1)
    {
        if (direction == 0)
        {
            int offset = 0;
            for (; offset + len < tot_len; offset += 2 * len)
            {
                int pnt1 = 0;
                int pnt2 = 0;
                while (pnt1 < len && pnt2 < len && offset + len + pnt2 < tot_len)
                {
                    if (merge_mem1[offset + pnt1] < merge_mem1[offset + len + pnt2])
                    {
                        merge_mem2[offset + pnt1 + pnt2] = merge_mem1[offset + pnt1];
                        pnt1++;
                    }
                    else
                    {
                        merge_mem2[offset + pnt1 + pnt2] = merge_mem1[offset + len + pnt2];
                        pnt2++;
                    }
                }
                while (pnt1 < len)
                {
                    merge_mem2[offset + pnt1 + pnt2] = merge_mem1[offset + pnt1];
                    pnt1++;
                }
                while (pnt2 < len && offset + len + pnt2 < tot_len)
                {
                    merge_mem2[offset + pnt1 + pnt2] = merge_mem1[offset + len + pnt2];
                    pnt2++;
                }
            }

            while (offset < tot_len)
            {
                merge_mem2[offset] = merge_mem1[offset];
                offset++;
            }
        }
        else
        {
            int offset = 0;
            for (; offset + len < tot_len; offset += 2 * len)
            {
                int pnt1 = 0;
                int pnt2 = 0;
                while (pnt1 < len && pnt2 < len && offset + len + pnt2 < tot_len)
                {
                    if (merge_mem2[offset + pnt1] < merge_mem2[offset + len + pnt2])
                    {
                        merge_mem1[offset + pnt1 + pnt2] = merge_mem2[offset + pnt1];
                        pnt1++;
                    }
                    else
                    {
                        merge_mem1[offset + pnt1 + pnt2] = merge_mem2[offset + len + pnt2];
                        pnt2++;
                    }
                }
                while (pnt1 < len)
                {
                    merge_mem1[offset + pnt1 + pnt2] = merge_mem2[offset + pnt1];
                    pnt1++;
                }
                while (pnt2 < len && offset + len + pnt2 < tot_len)
                {
                    merge_mem1[offset + pnt1 + pnt2] = merge_mem2[offset + len + pnt2];
                    pnt2++;
                }
            }

            while (offset < tot_len)
            {
                merge_mem1[offset] = merge_mem2[offset];
                offset++;
            }
        }
    }
    for (int i = 0; i < tot_len; i++)
    {
        if (direction == 1)
            as[i + l] = merge_mem2[i];
        else
            as[i + l] = merge_mem1[i];
    }
}

__kernel void merge(__global float * as,
                    __global float * bs,
                    const unsigned int n)
{
    unsigned int id = get_local_id(0);

    for (int offset = 0; offset < n; offset += MAX_WORK_SINGLE * WORKGROUP_SIZE)
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
    int direction = 0;
    for (int len = MAX_WORK_SINGLE; len < n; len <<= 1, direction ^= 1)
    {
        if (direction == 0)
        {
            int itemsPerWorkflow = 2 * len / WORKGROUP_SIZE;
            int offset = 0;
            for (; offset + len < n; offset += 2 * len)
            {
                int pre = id * itemsPerWorkflow;
                int l = max(0, pre - len) - 1, r = min(pre, len);
                while (l + 1 < r)
                {
                    int m = (l + r) / 2;
                    if (as[offset + m] < as[offset + len + pre - 1 - m])
                        l = m;
                    else
                        r = m;
                }

                ls[id] = offset + r, rs[id] = offset + len + pre - r;
                if (id == 0)
                {
                    ls[WORKGROUP_SIZE] = offset + len, rs[WORKGROUP_SIZE] = offset + 2 * len;
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

                barrier(CLK_GLOBAL_MEM_FENCE);

                for (int j = 0; j < itemsPerWorkflow; j++)
                {
                    as[offset + pre + j] = bs[offset + pre + j];
                }

                barrier(CLK_GLOBAL_MEM_FENCE);
            }

            if (id == 0)
            {
                while (offset < n)
                {
                    bs[offset] = as[offset];
                    offset++;
                }
            }

            barrier(CLK_GLOBAL_MEM_FENCE);
        }
        else
        {
            int itemsPerWorkflow = 2 * len / WORKGROUP_SIZE;
            int offset = 0;
            for (; offset + len < n; offset += 2 * len)
            {
                int pre = id * itemsPerWorkflow;
                int l = max(0, pre - len) - 1, r = min(pre, len);
                while (l + 1 < r)
                {
                    int m = (l + r) / 2;
                    if (bs[offset + m] < bs[offset + len + pre - 1 - m])
                        l = m;
                    else
                        r = m;
                }

                ls[id] = offset + r, rs[id] = offset + len + pre - r;
                if (id == 0)
                {
                    ls[WORKGROUP_SIZE] = offset + len, rs[WORKGROUP_SIZE] = offset + 2 * len;
                }

                barrier(CLK_LOCAL_MEM_FENCE);

                int pnt1 = 0, pnt2 = 0;
                while (ls[id] + pnt1 < ls[id + 1] && rs[id] + pnt2 < rs[id + 1])
                {
                    if (bs[ls[id] + pnt1] < bs[rs[id] + pnt2])
                    {
                        as[offset + pre + pnt1 + pnt2] = bs[ls[id] + pnt1];
                        pnt1++;
                    }
                    else
                    {
                        as[offset + pre + pnt1 + pnt2] = bs[rs[id] + pnt2];
                        pnt2++;
                    }
                }
                while (ls[id] + pnt1 < ls[id + 1])
                {
                    as[offset + pre + pnt1 + pnt2] = bs[ls[id] + pnt1];
                    pnt1++;
                }
                while (rs[id] + pnt2 < rs[id + 1])
                {
                    as[offset + pre + pnt1 + pnt2] = bs[rs[id] + pnt2];
                    pnt2++;
                }

                barrier(CLK_GLOBAL_MEM_FENCE);

                for (int j = 0; j < itemsPerWorkflow; j++)
                {
                    bs[offset + pre + j] = as[offset + pre + j];
                }

                barrier(CLK_GLOBAL_MEM_FENCE);
            }

            if (id == 0)
            {
                while (offset < n)
                {
                    as[offset] = bs[offset];
                    offset++;
                }
            }

            barrier(CLK_GLOBAL_MEM_FENCE);
        }
    }

    if (direction == 0)
    {
        for (int i = 0; i < n; i += WORKGROUP_SIZE)
        {
            if (i + id < n)
            {
                as[i + id] = bs[i + id];
            }
        }
    }
}