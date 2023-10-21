#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif


#line 6


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
