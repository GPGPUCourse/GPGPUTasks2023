// TODO
#line 2

__kernel void prefix_sum(__global const unsigned int *as, __global unsigned int *result, unsigned int n, unsigned int take_id) {

    unsigned int i = get_global_id(0);

    if (i < n && i & take_id) {
        result[i] += as[i / take_id - 1];
    }
}

__kernel void reduce(__global const unsigned int *as, __global unsigned int *bs, unsigned int n) {

    unsigned int i = get_global_id(0);

    if (i >= n){
        return;
    }

    unsigned int x = (2 * i >= n) ? 0 : as[2 * i];
    unsigned int y = (2 * i + 1 >= n) ? 0 : as[2 * i + 1];
    bs[i] = x + y;
}