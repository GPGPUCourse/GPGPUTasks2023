#line 2

unsigned int binsearch_right(__global const float *as, int left, int right, float x) {
    while(right - left > 1) {
        int n = (left + right) / 2;
        if (x < as[n]) {
            right = n;
        } else {
            left = n;
        }
    }
    return right;
}

unsigned int binsearch_left(__global const float *as, int left, int right, float x) {
    while(right - left > 1) {
        int n = (left + right) / 2;
        if (x <= as[n]) {
            right = n;
        } else {
            left = n;
        }
    }
    return right;
}

kernel void merge(global const float *sourceArray, __global float *destinationArray, const unsigned int arraySize, const unsigned int blockSize) {
    unsigned int index = get_global_id(0);

    if (index >= arraySize) {
        return;
    }

    float currentElement = sourceArray[index];

    int offsetInBlock = index % blockSize;
    int blockStartIndex = index - offsetInBlock;
    int blockEndIndex = blockStartIndex + blockSize;
    int blockNumber = blockStartIndex / blockSize;
    int isSecondBlock = blockNumber % 2;
    int neighborStartIndex = blockStartIndex + (isSecondBlock ? -blockSize : blockSize);
    int neighborEndIndex = blockEndIndex + (isSecondBlock ? -blockSize : blockSize);

    unsigned int position;

    if (isSecondBlock) {
        unsigned int neighborPosition = binsearch_right(sourceArray, neighborStartIndex - 1, neighborEndIndex, currentElement);
        position = offsetInBlock + neighborPosition;
    } else {
        unsigned int neighborPosition = binsearch_left(sourceArray, neighborStartIndex - 1, neighborEndIndex, currentElement);
        position = blockStartIndex + offsetInBlock + (neighborPosition - neighborStartIndex);
    }

    destinationArray[position] = currentElement;
}