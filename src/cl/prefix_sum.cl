__kernel void sweep_up(__global unsigned int *as, unsigned int n, unsigned int offset) {
    int id = get_global_id(0);
    int i = (id + 1) * (offset << 1) - 1;
    if (i < n) {
        as[i] += as[i - offset];
    }
}

__kernel void set_zero(__global unsigned int *as, unsigned int n) {
    int id = get_global_id(0);
    if (id == 0) {
        as[n - 1] = 0;
    }
}

__kernel void sweep_down(__global unsigned int *as, unsigned int n, unsigned int offset) {
    int id = get_global_id(0);
    int i = (id + 1) * (offset << 1) - 1;
    if (i < n) {
        unsigned int temp = as[i - offset];
        as[i - offset] = as[i];
        as[i] += temp;
    }
}

__kernel void shift_left(__global unsigned int *as, __global unsigned int *res, int n) {
    int id = get_global_id(0);
    if (id) {
        res[id - 1] = as[id];
    } else {
        res[n - 1] = as[n - 1];
    }
}