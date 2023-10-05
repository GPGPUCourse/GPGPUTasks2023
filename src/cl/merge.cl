inline bool comp(float a, float b, bool also_equal) {
    return also_equal ? a <= b : a < b;
}

inline unsigned int bin_search(unsigned int sorted_len, const float *data, float value, bool also_equal) {
    unsigned int search_start = 0;
    unsigned int search_end = sorted_len;
    while (search_start < search_end) {
        unsigned int middle = search_start + (search_end - search_start) / 2;
        // search in other half first element than greater (or equal)
        if (comp(value, data[middle], also_equal))
            search_end = middle;
        else
            search_start = middle + 1;
    }

    if (search_start < sorted_len && !comp(value, data[search_start], also_equal)) {
        search_start++;
    }

    return search_start;
}

__kernel void merge(__global const float *in, __global float *out, unsigned int n, unsigned int sorted_len) {
    unsigned int i = get_global_id(0);
    unsigned int my_block = (i / sorted_len);
    unsigned int other_block = my_block ^ 1;
    bool is_me_second = my_block % 2;

    __global const float *other_start = in + other_block * sorted_len;

    unsigned int my_pos = is_me_second ? i - sorted_len : i;
    my_pos += bin_search(sorted_len, other_start, in[i], is_me_second);
    out[my_pos] = in[i];
}
