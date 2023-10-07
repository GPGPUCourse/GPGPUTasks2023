#define ROUNDDOWN(a, b) (a) / (b) * (b)
#define ROUNDUP(a, b) ((a) + (b) - 1) / (b) * (b)

__kernel void merge(__global const float *src, __global float* dest, unsigned int n, unsigned int sorted_bs) {
    int id = get_global_id(0);
    if (id >= n) return;

    // src1 -- self block(containing worker), src2 -- other block
    int src1_bl_start = ROUNDDOWN(id, sorted_bs);
    int dest_bl_start = ROUNDDOWN(id, 2 * sorted_bs);
    bool is_right = (src1_bl_start == dest_bl_start);
    int src2_bl_start = src1_bl_start + (is_right ? sorted_bs : -sorted_bs);

    int tl = -1;
    int tr = sorted_bs;
    while (tr - tl > 1) {
        int tm = (tr + tl) / 2;
        if (src[id] > src[src2_bl_start + tm] || (src[id] == src[src2_bl_start + tm] && is_right)) {
            tl = tm;
        } else {
            tr = tm;
        }
    }

    dest[dest_bl_start + tr + (id - src1_bl_start)] = src[id];
}
