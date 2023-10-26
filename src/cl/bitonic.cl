__kernel void bitonicSortStep(__global unsigned *as, int blockToSortSize, int slidingBlockSize) {
    int blockToSortID = get_global_id(0) / (blockToSortSize / 2);
    int direction = blockToSortID % 2;
    int inBlockPos = get_global_id(0) % (blockToSortSize / 2);
    int i = blockToSortID * blockToSortSize + slidingBlockSize * (inBlockPos / (slidingBlockSize / 2)) + inBlockPos % (slidingBlockSize / 2);
    int j = i + slidingBlockSize / 2;
    unsigned x = as[i];
    unsigned y = as[j];
    if (!direction && x > y || direction && x < y) {
        as[i] = y;
        as[j] = x;
    }
}
