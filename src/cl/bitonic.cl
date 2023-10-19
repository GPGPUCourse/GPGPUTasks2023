__kernel void bitonic(__global float *a, const int big_window_size, const int small_window_size) {
    int gid = get_global_id(0);

    int small_half_window_size = small_window_size >> 1;
    int window_id = gid / small_half_window_size;
    int big_half_window_size = big_window_size >> 1;
    int big_window_id = gid / big_half_window_size;
    bool is_direct = big_window_id % 2 == 0;
    int up_i = (gid % small_half_window_size) + (window_id * small_window_size);
    int down_i = up_i + small_half_window_size;

    // printf("gid: %d\nup_i: %d\ndown_i: %d\n", gid, up_i, down_i);

    float up_elem = a[up_i];
    float down_elem = a[down_i];

    bool swap_direct = is_direct && (up_elem > down_elem);
    bool swap_backwards = (!is_direct) && (down_elem > up_elem);

    bool to_swap = swap_direct || swap_backwards;

    a[up_i] = to_swap ? down_elem : up_elem;
    a[down_i] = to_swap ? up_elem : down_elem;
}
