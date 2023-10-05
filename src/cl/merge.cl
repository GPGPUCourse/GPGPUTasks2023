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

int binary_search(__global float * as,
                  const unsigned int offset1, const unsigned int len1,
                  const unsigned int offset2, const unsigned int len2,
                  const unsigned int cnt)
{
    int l = cnt >= len2 ? cnt - len2 : 0;
    l--;

    int r = min(len1, cnt);

    while (l + 1 < r)
    {
        int m = (l + r) / 2;
        if (as[offset1 + m] < as[offset2 + cnt - 1 - m])
            l = m;
        else
            r = m;
    }

    return r;
}

void merge_simple(__global float * as,
                  __global float * bs,
                  unsigned int l1, const unsigned int l2,
                  unsigned int r1, const unsigned int r2,
                  unsigned int offset)
{
    while (l1 < l2 && r1 < r2)
    {
        if (as[l1] < as[r1])
        {
            bs[offset++] = as[l1++];
        }
        else
        {
            bs[offset++] = as[r1++];
        }
    }
    while (l1 < l2)
    {
        bs[offset++] = as[l1++];
    }
    while (r1 < r2)
    {
        bs[offset++] = as[r1++];
    }
}

// Сортировка на одной рабочей группе с бинарными поисками по диагоналям
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

    for (unsigned int len = MAX_WORK_SINGLE; len < n; len <<= 1)
    {
        unsigned int itemsPerWorkflow = 2 * len / WORKGROUP_SIZE;
        unsigned int offset = 0;

        for (; offset + len < n; offset += 2 * len)
        {
            unsigned int pre = id * itemsPerWorkflow;
            unsigned int l1, r1, l2, r2;
            if (offset + pre < n)
            {
                int r = binary_search(as, offset, len, offset + len, min(len, n - offset - len), pre);
                l1 = offset + r, r1 = offset + len + pre - r;
            }
            else
            {
                l1 = offset + len, r1 = n;
            }

            if (offset + pre + itemsPerWorkflow < n)
            {
                int r = binary_search(as, offset, len, offset + len, min(len, n - offset - len), pre + itemsPerWorkflow);
                l2 = offset + r, r2 = offset + len + pre + itemsPerWorkflow - r;
            }
            else
            {
                l2 = offset + len, r2 = n;
            }

            merge_simple(as, bs, l1, l2, r1, r2, offset + pre);
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

__kernel void sort_small_blocks(__global float * as,
                                const unsigned int n)
{
    unsigned int id = get_global_id(0);
    unsigned int l = id * MAX_WORK_SINGLE;
    unsigned int r = (id + 1) * MAX_WORK_SINGLE;

    if (r <= n) merge_not_parallel(as, l, r);
    else if (l < n) merge_not_parallel(as, l, n);
}

__kernel void merge(__global float * as,
                    __global float * bs,
                    const unsigned int len,
                    const unsigned int n)
{
    unsigned int gr = get_num_groups(0);
    unsigned int lid = get_local_id(0);
    unsigned int gid = get_group_id(0);

    unsigned int itemsPerWorkflow = 2 * len / WORKGROUP_SIZE;
    unsigned int offset = 0;
    for (; offset + 2 * gr * len <= n; offset += 2 * gr * len)
    {
        unsigned int offset2 = offset + gid * 2 * len;
        unsigned int r = binary_search(as, offset2, len, offset2 + len, len, itemsPerWorkflow * lid);
        unsigned int l1 = offset2 + r, r1 = offset2 + len + itemsPerWorkflow * lid - r;
        r = binary_search(as, offset2, len, offset2 + len, len, itemsPerWorkflow * (lid + 1));
        unsigned int l2 = offset2 + r, r2 = offset2 + len + itemsPerWorkflow * (lid + 1) - r;
        merge_simple(as, bs, l1, l2, r1, r2, offset2 + itemsPerWorkflow * lid);
    }
}

__kernel void merge_merge(__global float * as,
                          __global float * bs,
                          const unsigned int offset,
                          const unsigned int len,
                          const unsigned int n)
{
    unsigned int gr = get_num_groups(0);
    unsigned int lid = get_local_id(0);
    unsigned int gid = get_group_id(0);

    unsigned int itemsPerWorkGroup = 2 * len / gr;

    __local unsigned int l1, r1;
    if (lid == 0)
    {
        int r = binary_search(as, offset, len, offset + len, len,itemsPerWorkGroup * gid);
        l1 = offset + r, r1 = offset + len + itemsPerWorkGroup * gid - r;
    }
    __local unsigned int l2, r2;
    if (lid == 1)
    {
        int r = binary_search(as, offset, len, offset + len, len,itemsPerWorkGroup * (gid + 1));
        l2 = offset + r, r2 = offset + len + itemsPerWorkGroup * (gid + 1) - r;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned int itemsPerWorkflow = itemsPerWorkGroup / WORKGROUP_SIZE;

    unsigned int l3, r3, l4, r4;
    unsigned int r = binary_search(as, l1, l2 - l1, r1, r2 - r1, itemsPerWorkflow * lid);
    l3 = l1 + r, r3 = r1 + itemsPerWorkflow * lid - r;
    r = binary_search(as, l1, l2 - l1, r1, r2 - r1, itemsPerWorkflow * (lid + 1));
    l4 = l1 + r, r4 = r1 + itemsPerWorkflow * (lid + 1) - r;

    merge_simple(as, bs, l3, l4, r3, r4, offset + itemsPerWorkGroup * gid + itemsPerWorkflow * lid);
}
