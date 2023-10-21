__kernel void bitonic(__global float *as, unsigned n, unsigned sortingBlockSize, unsigned swapBlockSize) {
    unsigned idx = get_global_id(0);
    unsigned sortingBlockId = idx / (sortingBlockSize / 2);
    unsigned idInSortingBlock = idx % (sortingBlockSize / 2);
    unsigned swapBlockId = idInSortingBlock / (swapBlockSize / 2);
    unsigned idInSwapBlock = idInSortingBlock % (swapBlockSize / 2);
    unsigned pos = sortingBlockId * sortingBlockSize + swapBlockId * swapBlockSize + idInSwapBlock;
    unsigned neighborPos = pos + swapBlockSize / 2;
    bool inverse = sortingBlockId & 1;
    if (pos >= n || neighborPos >= n)
        return;
//    printf("SortingBlockId: %d, idInSortingBlock: %d, swapBlockId: %d, idInSwapBlock: %d, inverse: %d, pos: %d, neighborPos: %d\n", sortingBlockId, idInSortingBlock, swapBlockId, idInSwapBlock, inverse, pos, neighborPos);
    float x = as[pos];
    float y = as[neighborPos];
    if ((!inverse && x > y) || (inverse && x < y)) {
        as[pos] = y;
        as[neighborPos] = x;
//        printf("as[%d]: %.2f, as[%d]: %.2f (SWAP)\n", pos, x, neighborPos, y);
    }
    else {
//        printf("as[%d]: %.2f, as[%d]: %.2f\n", pos, x, neighborPos, y);
    }
}
