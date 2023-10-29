#ifdef WORK_GROUP_SIZE
#ifdef BLOCK_BITS

#define BLOCK_NUMBERS (1 << BLOCK_BITS)

#define GET_VALUE(raw_val, shift) (((raw_val) >> (shift)) & ((1 << BLOCK_BITS) - 1))

__kernel void radix(
    __global unsigned int *as,
    unsigned n,
    __global unsigned *counts,
    __global unsigned *countsPrefSum,
    __global unsigned *bs,
    unsigned shift
    ) {

    unsigned globalId = get_global_id(0);
    unsigned localId = get_local_id(0);
    unsigned wgId = get_group_id(0);
    unsigned fullValue = as[globalId];
    unsigned value = GET_VALUE(fullValue, shift);


    unsigned resultPos = localId;
    unsigned prefIdx = value * n / WORK_GROUP_SIZE + wgId;
//    printf("gId: %d, val: %d, prefIdx: %d\n", globalId, value, prefIdx);
    if (prefIdx > 0)
        resultPos += countsPrefSum[prefIdx - 1];
//    printf("(%d) prefIdx: %d, starting resultPos: %d, wg: %d\n", globalId, prefIdx, resultPos, wgId);
    for (int i = 0; i < BLOCK_NUMBERS - 1; i++) {
        resultPos -= (unsigned)(i < value) * counts[wgId * BLOCK_NUMBERS + i];
    }
//    printf("Putting %d from pos %d to pos %d\n", fullValue, globalId, resultPos);
    bs[resultPos] = fullValue;
}

// n must be a multiple of WORK_GROUP_SIZE
// work group size must be larget than
__kernel void calculate_counts_in_each_block(
    __global const unsigned *as,
    unsigned n,
    __global unsigned *counts,
    unsigned shift)
{
    unsigned globalId = get_global_id(0);
    unsigned wgId = get_group_id(0);

    if (n % WORK_GROUP_SIZE != 0)
        printf("n=%d is not a multiple of WORK_GROUP_SIZE=%d\n", n, WORK_GROUP_SIZE);
    unsigned fullValue = as[globalId];
    unsigned value = GET_VALUE(fullValue, shift);
//    printf("WG: %d, value = %d, fullValue = %d, shift = %d, BLOCK_BITS = %d\n", wgId, value, fullValue, shift, BLOCK_BITS);
    __local unsigned local_counts[BLOCK_NUMBERS];
    barrier(CLK_LOCAL_MEM_FENCE);
    unsigned localId = get_local_id(0);
    if (get_local_size(0) < BLOCK_NUMBERS)
        printf("local_size=%d is less than BLOCK_NUMBERS=%d\n", get_local_size(0), BLOCK_NUMBERS);
    if (localId < BLOCK_NUMBERS)
        local_counts[localId] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);
    atomic_add(local_counts + value, 1);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId < BLOCK_NUMBERS)
        counts[wgId * BLOCK_NUMBERS + localId] = local_counts[localId];
}

// n must be a multiple of WORK_GROUP_SIZE
__kernel void sort_for_each_work_group(
    __global unsigned *as,
    unsigned n,
    unsigned shift)
{
    if (n % WORK_GROUP_SIZE != 0)
        printf("n=%d is not a multiple of WORK_GROUP_SIZE=%d\n", n, WORK_GROUP_SIZE);

    unsigned localId = get_local_id(0);
    unsigned globalId = get_global_id(0);
    unsigned groupId = get_group_id(0);

    if (globalId >= n)
        printf("Oh no...\n");

//    printf("as[%d] == %d\n", globalId, as[globalId]);


    __local unsigned localArr[2][WORK_GROUP_SIZE];
    localArr[0][localId] = as[globalId];
//    printf("WG: %d, localArr[%d] = as[%d] = %d = %d\n", groupId, localId, globalId, as[globalId], localArr[localId]);
//    barrier(CLK_LOCAL_MEM_FENCE);
//    printf("WG: %d, localArr = [%d %d %d %d]\n", groupId, localArr[0], localArr[1], localArr[2], localArr[3]);
    barrier(CLK_LOCAL_MEM_FENCE);

    if (localId == 0) {
        unsigned s = 0;
        for (unsigned sortingBlockSize = 2; sortingBlockSize <= WORK_GROUP_SIZE; sortingBlockSize <<= 1) {
            unsigned *source = &localArr[s];
//            printf("Sorting block size: %d\n", sortingBlockSize);
//            printf("[\n\t");
//            for (int t = 0; t < WORK_GROUP_SIZE; t++) {
//                printf("%d ", source[t]);
//            }
//            printf("\n\t");
//            for (int t = 0; t < WORK_GROUP_SIZE; t++) {
//                printf("%d ", GET_VALUE(source[t], shift));
//            }
//            printf("\n]\n");
            unsigned *dest = &localArr[s ^ 1];
            for (unsigned lBegin = 0; lBegin < WORK_GROUP_SIZE; lBegin += sortingBlockSize) {
                unsigned rBegin = lBegin + sortingBlockSize / 2;
                unsigned i = lBegin, j = rBegin, k = lBegin;
                while (i < rBegin && j < lBegin + sortingBlockSize) {
                    if (GET_VALUE(source[i], shift) <= GET_VALUE(source[j], shift))
                        dest[k++] = source[i++];
                    else
                        dest[k++] = source[j++];
                }
                while (i < rBegin)
                    dest[k++] = source[i++];
                while (j < lBegin + sortingBlockSize)
                    dest[k++] = source[j++];
            }
//            printf("Result: %d\n", sortingBlockSize);
//            printf("[\n\t");
//            for (int t = 0; t < WORK_GROUP_SIZE; t++) {
//                printf("%d ", dest[t]);
//            }
//            printf("\n\t");
//            for (int t = 0; t < WORK_GROUP_SIZE; t++) {
//                printf("%d ", GET_VALUE(dest[t], shift));
//            }
//            printf("\n]\n");
            s ^= 1;
        }
        for (int i = 0; i < WORK_GROUP_SIZE; i++) {
            as[WORK_GROUP_SIZE * groupId + i] = localArr[s][i];
//            printf("as[%d] = localArr[%d][%d] = %d\n", WORK_GROUP_SIZE * groupId + i, s, localId, localArr[s][i]);

        }
    }
}




#endif
#endif
