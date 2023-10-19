
__kernel void bitonic_step(__global float *as, int n, int sort_len, int cmp_dist) {
    int id = get_global_id(0);
    //TODO use n
    int i;
    {
        int pairs_block_start = id / cmp_dist * cmp_dist;
        i = pairs_block_start * 2 + id % cmp_dist;
    }
    int j = i + cmp_dist;

    float x = as[i];
    float y = as[j];
    bool asc_dir = (i / sort_len) % 2 == 0;
    if (asc_dir && x > y || !asc_dir && x < y) {
        as[i] = y;
        as[j] = x;
    }
}




