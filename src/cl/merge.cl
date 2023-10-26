inline bool comp(uint a, uint b, bool also_equal, uint shift, uint k) {
    uint mask_a = (a >> shift) & ((1 << k) - 1);
    uint mask_b = (b >> shift) & ((1 << k) - 1);
    return also_equal ? mask_a <= mask_b : mask_a < mask_b;
}

inline unsigned int bin_search(unsigned int sorted_len, __global const uint *data, uint value, bool also_equal, uint shift, uint k) {
    unsigned int search_start = 0;
    unsigned int search_end = sorted_len;
    while (search_start < search_end) {
        unsigned int middle = search_start + (search_end - search_start) / 2;
        // search in other half first element than greater (or equal)
        if (comp(value, data[middle], also_equal, shift, k))
            search_end = middle;
        else
            search_start = middle + 1;
    }

    if (search_start < sorted_len && !comp(value, data[search_start], also_equal, shift, k)) {
        search_start++;
    }

    return search_start;
}

__kernel void merge(__global const uint *in, __global uint *out, unsigned int n, unsigned int sorted_len, uint shift, uint k) {
    unsigned int i = get_global_id(0);
    unsigned int my_block = (i / sorted_len);
    unsigned int other_block = my_block ^ 1;
    bool is_me_second = my_block % 2;

    __global const uint *other_start = in + other_block * sorted_len;

    unsigned int my_pos = is_me_second ? i - sorted_len : i;
    my_pos += bin_search(sorted_len, other_start, in[i], !is_me_second, shift, k);
    out[my_pos] = in[i];
}
