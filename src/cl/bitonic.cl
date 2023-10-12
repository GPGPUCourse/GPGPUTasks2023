#define max(x, y) x > y ? x : y
#define min(x, y) x < y ? x : y

__kernel void bitonic(__global float *as, const unsigned len) {
    unsigned gid = get_global_id(0);
    unsigned point = len;

    while (point) {
        gid = get_global_id(0);
        unsigned decreasing = (2 * gid / len) % 2;
        gid = (2 * gid) / point * point + gid % (point / 2);
        point = point / 2;
        float max = max(as[gid], as[gid + point]);
        float min = min(as[gid], as[gid + point]);
        if (decreasing) {
            as[gid] = max;
            as[gid + point] = min;
        } else {
            as[gid] = min;
            as[gid + point] = max;
        }

        point /= 2;

        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
