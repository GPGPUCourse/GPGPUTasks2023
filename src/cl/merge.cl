
__kernel void merge(const __global float *as_gpu, __global float* buffer, const uint mergeSize, const uint n) {
    const uint idx = get_global_id(0);
    if (idx >= n) {
        return;
    }

    const uint mergeGroupId = idx / mergeSize;
    const uint groupElementId = idx % mergeSize;

    // Even (start count from 0) blocks I menally position vertical
    const bool isVertical = mergeGroupId % 2 == 0;
    const uint pairedGroupId = isVertical ? mergeGroupId + 1 : mergeGroupId - 1;

    int l = pairedGroupId * mergeSize;  // start index of pairedGroup
    int r = min(l + mergeSize, n);      // index pointing to first element after pairedGroup
    while (l < r) {
        int mid = l + (r - l) / 2;
        if (as_gpu[mid] > as_gpu[idx] || (isVertical && as_gpu[mid] == as_gpu[idx])) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }

    const uint bufferStartIdx = min(mergeGroupId, pairedGroupId) * mergeSize;
    const uint bufferIdx = bufferStartIdx + groupElementId + (r - pairedGroupId * mergeSize);
    buffer[bufferIdx] = as_gpu[idx];
}