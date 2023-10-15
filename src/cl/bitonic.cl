#define max(x, y) x > y ? x : y
#define min(x, y) x < y ? x : y

__kernel void bitonic(__global float *as, const unsigned window, const unsigned arrow, const unsigned size) {
    unsigned id = get_global_id(0);
    unsigned id2x = id << 1;

    if (id2x >= size) {
        return;
    }

    unsigned start = (id2x / window);
    int decreasing = start % 2;

    start = start * window;
    unsigned arrow2x = arrow << 1;
    id = start + id % arrow + (id2x - start) / arrow2x * arrow2x;

    float max = max(as[id], as[id + arrow]);
    float min = min(as[id], as[id + arrow]);

    as[id] = (decreasing) ? max : min;
    as[id + arrow] = (decreasing) ? min : max;
}
