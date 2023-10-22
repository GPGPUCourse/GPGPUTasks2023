__kernel void bitonic(__global float *as, const unsigned int block_size, const unsigned int step) {
    const int id = get_global_id(0);
    int fst_ind = 2 * id - id % step;
    int snd_ind = fst_ind + step;
    int direction = ((id / block_size) % 2) * 2 - 1;
    
    if (as[fst_ind] * direction < as[snd_ind] * direction) {
        float tmp = as[fst_ind];
        as[fst_ind] = as[snd_ind];
        as[snd_ind] = tmp;
    }
}