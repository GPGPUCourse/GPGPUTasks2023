kernel void bitonic(global float *as, unsigned b, unsigned k) {
    unsigned id = get_global_id(0);

    unsigned i = 2 * k * (id / k) + id % k; // it just works
    unsigned j = i + k;

    // if in blue block
    bool blue = (id / (1 << (b - 1))) % 2 == 0;

    if ((blue && as[i] > as[j]) || (!blue && as[i] < as[j])) {
        float tmp = as[i];
        as[i] = as[j];
        as[j] = tmp;
    }
}
