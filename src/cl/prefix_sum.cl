kernel void reduce(global unsigned *as, unsigned k) {
    unsigned id = get_global_id(0);
    unsigned index = id + 1;
    // update cell if not first iteration and (id + 1) is divisible by 2**k
    unsigned shift = index & ((1 << k) - 1);
    if (k && !shift) {
        as[id] += as[id - (1 << (k - 1))];
    }
}
kernel void prefix(global const unsigned *as, global unsigned *result, unsigned k) {
    unsigned id = get_global_id(0);
    unsigned index = id + 1;
    unsigned shift = index & ((1 << k) - 1);
    // get the cell if we need it
    if ((index >> k) & 1) {
        result[id] += as[id - shift];
    }
}
