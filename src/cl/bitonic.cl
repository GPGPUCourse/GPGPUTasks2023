#define swap(x, y) {\
    float z = x; \
    x = y; \
    y = z; } \


#define cas(expected, x, y) \
    if ((x < y) ^ expected) { \ 
        swap(x, y); \
    }

__kernel void bitonic(
    __global float *as, 
    int blockToSortSize, 
    int partialSize
) {
    int pairsInBlock = blockToSortSize >> 1;
    int pairsPartial = partialSize >> 1;
    int blockToSortID = get_global_id(0) / pairsInBlock;
    int orientationInPair = blockToSortID & 1;
    int inBlockPos = get_global_id(0) % pairsInBlock;
    int i = blockToSortID * blockToSortSize + partialSize * (inBlockPos / pairsPartial) + inBlockPos % pairsPartial;
    int j = i + pairsPartial;
    cas(!orientationInPair, as[i], as[j]);
    cas(orientationInPair, as[j], as[i]);
}
