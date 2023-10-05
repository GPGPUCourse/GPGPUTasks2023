#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define WORKGROUP_SIZE 128
#define MAX_WORK_SINGLE 256
#define MAX_WORKGROUPS 1024

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
    int l = cnt >= len1 ? cnt - len1 : 0;
    if (l + len2 < cnt) l = cnt - len2;
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
                int r = binary_search(as, offset, len, offset + len, min(len, n - offset - len),
                                      pre + itemsPerWorkflow);
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
    unsigned int id = get_global_id(0);
    unsigned int gr = get_num_groups(0);
    unsigned int lid = get_local_id(0);
    unsigned int gid = get_group_id(0);

    unsigned int offset = 0;
    for (; offset + 2 * gr * len <= n; offset += 2 * gr * len)
    {
        if (lid != 0)
            continue;

        unsigned int le = offset + gid * 2 * len;
        unsigned int pnt1 = 0, pnt2 = 0;
        while (pnt1 < len && pnt2 < len)
        {
            if (as[le + pnt1] < as[le + len + pnt2])
            {
                bs[le + pnt1 + pnt2] = as[le + pnt1];
                pnt1++;
            }
            else
            {
                bs[le + pnt1 + pnt2] = as[le + len + pnt2];
                pnt2++;
            }
        }
        while (pnt1 < len)
        {
            bs[le + pnt1 + pnt2] = as[le + pnt1];
            pnt1++;
        }
        while (pnt2 < len)
        {
            bs[le + pnt1 + pnt2] = as[le + len + pnt2];
            pnt2++;
        }
    }
}

__kernel void merge_merge_prepare(__global float * as,
                                  __global unsigned int * ls,
                                  __global unsigned int * rs,
                                  const unsigned int offset,
                                  const unsigned int len,
                                  const unsigned int n)
{
    unsigned int id = get_global_id(0);
    unsigned int gr = get_num_groups(0);
    unsigned int lid = get_local_id(0);
    unsigned int gid = get_group_id(0);

    unsigned int itemsPerWorkGroup = 2 * len / gr;

    if (lid == 0)
    {
        unsigned int pre = gid * itemsPerWorkGroup;

        int l = pre > len ? pre - len : 0;
        if (l + (n - offset - len) < pre)
            l = pre - (n - offset - len);
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
        ls[gid] = offset + r, rs[gid] = offset + len + pre - r;

        if (gid == 0)
        {
            ls[gr] = offset + len, rs[gr] = offset + 2 * len;
        }
    }
}

__kernel void merge_merge(__global float * as,
                          __global float * bs,
                          __global unsigned int * ls,
                          __global unsigned int * rs,
                          const unsigned int offset,
                          const unsigned int len,
                          const unsigned int n)
{
    unsigned int id = get_global_id(0);
    unsigned int gr = get_num_groups(0);
    unsigned int lid = get_local_id(0);
    unsigned int gid = get_group_id(0);

    printf("%d\n", rs[gid + 1] - rs[gid] + ls[gid + 1] - ls[gid]);
}
