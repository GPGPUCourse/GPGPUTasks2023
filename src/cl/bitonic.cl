// TODO
#line 2

__kernel void bitonic(__global float *as, unsigned int n, unsigned int block_size, unsigned int gap) {

    unsigned int low_i = get_global_id(0);
    unsigned int high_i = low_i ^ gap;

    if (high_i > low_i) {
        float low = as[low_i];
        float high = as[high_i];
        if (low_i & block_size) {
            if (low < high) {
                as[low_i] = high;
                as[high_i] = low;
            }
        } else {
            if (low > high) {
                as[low_i] = high;
                as[high_i] = low;
            }
        }
    }
}
