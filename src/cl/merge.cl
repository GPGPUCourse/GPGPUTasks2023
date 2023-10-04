#define LOCAL_SIZE 128
__kernel void merge_local(__global const float *in, __global float *out, int n) {
    __local float aux[LOCAL_SIZE];
    const int global_i = get_global_id(0);
    const int i = get_local_id(0);// index in workgroup
    if (global_i < n) {
        // Load block in AUX[WG]
        aux[i] = in[global_i];
        barrier(CLK_LOCAL_MEM_FENCE);// make sure AUX is entirely up to date
        // Now we will merge sub-sequences of length 1,2,...,WG/2
        for (int sorted = 1; sorted < LOCAL_SIZE; sorted <<= 1) {
            const float iData = aux[i];
            const int li = i & (sorted - 1);      // index in our sequence in 0..length-1
            const int sibling = (i - li) ^ sorted;// beginning of the sibling sequence
            int pos = 0;
            for (int inc = sorted; inc != 0; inc /= 2) {// increment for dichotomic search
                const int j = sibling + pos + inc - 1;
                const float jData = aux[j];
                bool smaller = (jData < iData) || (jData == iData && j < i);
                pos += (smaller) ? inc : 0;
                pos = min(pos, sorted);
            }
            const int bits = 2 * sorted - 1;                   // mask for destination
            const int dest = ((li + pos) & bits) | (i & ~bits);// destination index in merged sequence
            barrier(CLK_LOCAL_MEM_FENCE);
            aux[dest] = iData;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        // Write output
        out[global_i] = aux[i];
    }
}

__kernel void merge(__global const float *a, __global float *b, const unsigned int sorted, const unsigned int n) {
    const int i = get_global_id(0);
    if (i < n) {
        const unsigned int first_offset = (i / (2 * sorted)) * 2 * sorted;
        const unsigned int second_offset = first_offset + sorted;
        const unsigned int second_size = min(sorted, n - second_offset);
        if (second_offset >= n) {
            b[i] = a[i];
            return;
        }
        const int is_left = i < second_offset;
        int lower = is_left ? second_offset : first_offset;
        int upper = is_left ? (second_offset + second_size) : second_offset;
        const float item = a[i];
        while (lower < upper) {
            int m = (lower + upper) / 2;
            if (a[m] < item) {
                lower = m + 1;
            } else {
                upper = m;
            }
        }
        upper = lower;
        while (upper < second_offset && a[upper] == item) {
            ++upper;
        }
        const int left_offset = is_left ? i : upper;
        const int right_offset = (is_left ? lower : i) - second_offset;
        b[left_offset + right_offset] = item;
    }
}