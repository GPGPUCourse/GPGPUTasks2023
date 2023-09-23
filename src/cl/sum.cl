// TODO


kernel void simple(global const unsigned *a, global unsigned *sum, unsigned n) {
    int id = get_global_id(0);
    atomic_add(sum, a[id]);
}

#define valuesPerItem 64
kernel void cycle(global const unsigned *a, global unsigned *sum, unsigned n) {
    int gid = get_global_id(0);
    unsigned s = 0;
    for (int i = 0; i < valuesPerItem; ++i) {
        unsigned id = gid * valuesPerItem + i;
        if (id < n) s += a[id];
    }
    atomic_add(sum, s);
}

kernel void cycle_break(global const unsigned *a, global unsigned *sum, unsigned n) {
    int gid = get_global_id(0);
    unsigned s = 0;
    for (int i = 0; i < valuesPerItem; ++i) {
        unsigned id = gid * valuesPerItem + i;
        if (id < n) s += a[id];
        else break;
    }
    atomic_add(sum, s);
}

kernel void smart_cycle(global const unsigned *a, global unsigned *sum, unsigned n) {
    int lid = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0);

    unsigned s = 0;
    for (int i = 0; i < valuesPerItem; ++i) {
        unsigned id = group_id * group_size * valuesPerItem + i * group_size + lid;
        if (id < n) s += a[id];
    }
    atomic_add(sum, s);
}

#define groupSize 128
kernel void local_mem(global const unsigned *a, global unsigned *sum, unsigned n) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);

    local unsigned buf[groupSize];
    buf[lid] = gid < n ? a[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        unsigned s = 0;
        for (int i = 0; i < groupSize; ++i) {
            s += buf[i];
        }
        atomic_add(sum, s);
    }
}

kernel void tree(global const unsigned *a, global unsigned *sum, unsigned n) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);

    local unsigned buf[groupSize];
    buf[lid] = gid < n ? a[gid] : 0;

    barrier(CLK_LOCAL_MEM_FENCE);
    for (int num = groupSize; num > 1; num /= 2) {
        if (lid * 2 < num) buf[lid] += buf[lid + num / 2];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) atomic_add(sum, buf[0]);
}
