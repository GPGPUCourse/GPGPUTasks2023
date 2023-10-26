__kernel void mergesortPhase(__global const float *from, __global float *to, unsigned n, int blockLength) {
    int index = get_global_id(0);
    if (index >= n)
        return;
    int blockIndex = index / blockLength;
    int blockBegin = blockIndex * blockLength;
    int blockSide = blockIndex % 2;
    int inBlockPosition = index % blockLength;
    __global const float *block = from + blockBegin;
    float this = block[inBlockPosition];
    int otherBlockBegin = blockBegin + (blockSide == 0 ? +blockLength : -blockLength);
    __global const float *otherBlock = from + otherBlockBegin;
    int left = -1;
    int right = blockLength;
    while (right - left > 1) {
        int mid = (left + right) / 2;
        float other = otherBlock[mid];
        if (other < this || other == this && blockSide)
            left = mid;
        else
            right = mid;
    }
    int newBlockBegin = (blockSide == 0 ? blockBegin : otherBlockBegin);
    to[newBlockBegin + inBlockPosition + left + 1] = this;
}

#define K 128

/// \pre blockLength <= K / 2
__kernel void mergesortPhaseLocal(__global const float *from, __global float *to, unsigned n, int blockLength) {
    __local float localFrom[K];
    int index = get_global_id(0);
    int localIndex = get_local_id(0);
    float this = from[index];
    localFrom[localIndex] = this;
    barrier(CLK_LOCAL_MEM_FENCE);
    int blockIndex = localIndex / blockLength;
    int blockSide = blockIndex % 2;
    int blockBegin = blockIndex * blockLength;
    int otherBlockBegin = blockBegin + (blockSide == 0 ? +blockLength : -blockLength);
    int left = -1;
    int right = blockLength;
    while (right - left > 1) {
        int mid = (left + right) / 2;
        float other = localFrom[otherBlockBegin + mid];
        if (other < this || other == this && blockSide)
            left = mid;
        else
            right = mid;
    }
    __local float localTo[K];
    int newBlockBegin = (blockSide == 0 ? blockBegin : otherBlockBegin);
    localTo[newBlockBegin + localIndex % blockLength + left + 1] = this;
    barrier(CLK_LOCAL_MEM_FENCE);
    to[index] = localTo[localIndex];
}

///------------------
/// Diagonal method
///------------------

#define WORK_GROUP_SIZE 64

struct BlocksShift {
    int shift0;
    int shift1;
};

struct BlocksShift getBlocksShift(
    __global const float *block0,
    __global const float *block1,
    int blockLength,
    int workItemShift
) {
    int intersectionPoint;
    {
        int left = -1;
        if (workItemShift - 1 - left > blockLength)
            left = workItemShift - blockLength - 1;
        int right = workItemShift;
        if (right > blockLength)
            right = blockLength;
        while (right - left > 1) {
            int mid = (left + right) / 2;
            if (block0[mid] <= block1[workItemShift - 1 - mid]) {
                left = mid;
            }
            else {
                right = mid;
            }
        }
        intersectionPoint = left + 1;
    }
    struct BlocksShift ret = { intersectionPoint, workItemShift - intersectionPoint };
    return ret;
}

/// \pre blockLength >= K
/// \pre blockLength % K == 0
__kernel void mergesortDiagonalPhase(__global const float *from, __global float *to, unsigned n, int blockLength) {
    int workItemBegin = get_global_id(0) * K;
    int blockIndex = workItemBegin / blockLength;
    if (blockIndex % 2 == 1)
        --blockIndex;
    int blockBegin = blockIndex * blockLength;
    __global const float *block0 = from + blockBegin;
    __global const float *block1 = block0 + blockLength;
    
    int workItemShift = workItemBegin - blockBegin;
    struct BlocksShift blocksShift = getBlocksShift(block0, block1, blockLength, workItemShift);
    int i = blocksShift.shift0;
    int j = blocksShift.shift1;

    __local float local_to[K * WORK_GROUP_SIZE];
    int localWorkItemIndex = get_local_id(0);
    int localWorkItemBegin = localWorkItemIndex * K;
    for (int iter = 0; iter < K; ++iter) {
        float a0 = i < blockLength ? block0[i] : 0;
        float a1 = j < blockLength ? block1[j] : 0;
        if (j >= blockLength || i < blockLength && j < blockLength && a0 <= a1) {
            local_to[localWorkItemBegin + iter] = a0;
            ++i;
        } else {
            local_to[localWorkItemBegin + iter] = a1;
            ++j;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    int groupIndex = get_group_id(0);
    int groupBegin = groupIndex * WORK_GROUP_SIZE * K;
    for (int i = 0; i < K; ++i)
        to[groupBegin + i * WORK_GROUP_SIZE + localWorkItemIndex] = local_to[i * WORK_GROUP_SIZE + localWorkItemIndex];
}
