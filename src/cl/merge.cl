__kernel void mergeSortStep(__global const float* source,
                            __global float* dest,
                            unsigned int n,
                            unsigned int blockSize) {
    unsigned int index = get_global_id(0);
    if (index >= n)
        return;
    unsigned int currentBlockIndex = index / blockSize;
    unsigned int pairedBlockIndex = currentBlockIndex ^ 1;
    __global const float *pairedBlock = source + pairedBlockIndex * blockSize;
    float val = source[index];
    int l = -1;
    int r = blockSize;
    while (r - l > 1) {
        int m = (l + r) / 2;
        if (pairedBlock[m] > val || (pairedBlock[m] == val && (currentBlockIndex & 1)))
            r = m;
        else
            l = m;
    }
    unsigned int indexInBlock = index % blockSize;
    unsigned int destBlockBegin = min(currentBlockIndex, pairedBlockIndex) * blockSize;
    unsigned int itemsInPairedLowerThanValue = r;
    dest[destBlockBegin + itemsInPairedLowerThanValue + indexInBlock] = val;
}