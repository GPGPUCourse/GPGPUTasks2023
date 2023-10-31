kernel void reduce(global unsigned *result, global const unsigned *a, unsigned k) {
    unsigned id = get_global_id(0);
    unsigned base = (1u << k) * id;
    result[base] = a[base];
    result[base] += a[base + (1u << k - 1)];
}

kernel void pick(global unsigned *result, global const unsigned *a, unsigned k) {
    unsigned id = get_global_id(0) + 1;
    // get the cell if we need it
    if (id >> k & 1u)
        result[id - 1] += a[id >> k + 1 << k + 1];
}
