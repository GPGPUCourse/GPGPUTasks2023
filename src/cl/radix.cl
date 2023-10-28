#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6

#define get_elem(e, offset) (((e) >> (offset)) & ((B) -1))

// B = 2^k, 0 <= a[i] < B.
__kernel void build_count(const __global unsigned *as, const unsigned offset, __global unsigned *count) {
    unsigned gid = get_global_id(0);
    unsigned lid = get_local_id(0);

    __local unsigned cnt[B];
    unsigned start = get_group_id(0) * B;
    unsigned shift = get_local_size(0);
    for (int i = lid; i < B; i += shift) {
        cnt[i] = 0;
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    unsigned e = get_elem(as[gid], offset);
    atomic_add(&cnt[e], 1);

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = lid; i < B; i += shift) {
        count[start + i] = cnt[i];
    }
}

__kernel void dummy_prefix_sums(__global unsigned *in, __global unsigned *out, int resp) {
    const int gid = get_global_id(0);

    if (gid % resp == 0) {
        unsigned accum = 0;
        for (int i = gid; i < gid + resp; ++i) {
            accum += in[i];
            out[i] = accum;
        }
    }
}

// (m, k) -> (k, m)
// (!) TILE_SIZE | m, k
__kernel void transpose(__global const unsigned *a, __global unsigned *res, unsigned m, unsigned k) {
    const int i = get_global_id(1);
    const int j = get_global_id(0);

    __local float tile[TILE_SIZE][TILE_SIZE];

    const int li = get_local_id(1);
    const int lj = get_local_id(0);
    tile[lj][li] = a[i * k + j];

    barrier(CLK_LOCAL_MEM_FENCE);

    const int gi = get_group_id(1);
    const int gj = get_group_id(0);

    res[gj * TILE_SIZE * m + gi * TILE_SIZE + li * m + lj] = tile[li][lj];
}

__kernel void prefix_sum_sparse(const __global unsigned *sparse_in, __global unsigned *sparse_out, int sz, int n) {
    int i = get_global_id(0);
    sparse_out[i] = sparse_in[i];
    if (i + sz < n) {
        sparse_out[i] += sparse_in[i + sz];
    }
}

__kernel void prefix_sum_supplement(__global unsigned *sparse, __global unsigned *res, int sz) {
    int i = get_global_id(0);
    int j = (i + 1) % (2 * sz);
    if (sz == 1) {
        res[i] = 0;
    }
    if (j >= sz) {
        res[i] += sparse[j - sz];
    }
}

inline bool comp(unsigned e1, unsigned e2, bool weak) {
    return weak ? e1 < e2 : e1 <= e2;
}

int binary_search(const __local unsigned *a, int n, unsigned e, bool weak) {
    int lo = -1;
    int hi = n;
    while (hi - lo > 1) {
        int mid = (lo + hi) / 2;
        if (comp(a[mid], e, weak)) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return lo + 1;
}

__kernel void merge(const __global unsigned *in, __global unsigned *out, const int sorted, unsigned offset) {
    const int i = get_global_id(0);

    unsigned elem = in[i];

    __local unsigned copy[WG_SIZE];
    copy[get_local_id(0)] = get_elem(elem, offset);
    barrier(CLK_LOCAL_MEM_FENCE);

    const int block_begin = (i / (2 * sorted)) * (2 * sorted);
    bool at_left = (i % (2 * sorted)) < sorted;

    const int l = block_begin % WG_SIZE + at_left * sorted;

    const int less = binary_search(copy + l, sorted, get_elem(elem, offset), at_left);
    out[i - sorted * !at_left + less] = elem;
}

__kernel void radix_sort(const __global unsigned *in, __global unsigned *out, const __global unsigned *pf,
                         const unsigned work_group_cnt, const unsigned offset, const __global unsigned *count_pf) {
    const int gid = get_global_id(0);

    unsigned elem = in[gid];
    unsigned e = get_elem(elem, offset);

    const int pf_index = work_group_cnt * e + get_group_id(0) - 1;

    int lesser = ((pf_index >= 0) ? pf[pf_index] : 0) + (gid - gid / WG_SIZE * WG_SIZE) -
                 ((e > 0) ? count_pf[get_group_id(0) * B + e - 1] : 0);

    out[lesser] = elem;
}
