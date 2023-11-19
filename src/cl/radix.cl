#define WORK_SIZE 64
#define UNIQUE_VALS_COUNT 4
#define MASK (UNIQUE_VALS_COUNT - 1)

__kernel void radix_count(const __global uint *as, __global uint *counters, __global uint *counters_temp,
                          const uint shiftR, const uint n) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint groupId = get_group_id(0);
    const uint wgNum = get_num_groups(0);

    if (gid >= n) {
        return;
    }

    // Empty local storage
    __local uint cnts[UNIQUE_VALS_COUNT];
    if (lid < UNIQUE_VALS_COUNT) {
        cnts[lid] = 0;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Atomically increase counters
    if (gid < n) {
        const uint val = (as[gid] >> shiftR) & MASK;
        atomic_inc(&cnts[val]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Push data back into memory (already transposed)
    if (lid < UNIQUE_VALS_COUNT) {
        counters[lid * wgNum + groupId] = cnts[lid];
        counters_temp[lid * wgNum + groupId] = cnts[lid];
    }
}

__kernel void prefix(__global uint *as, __global uint *result, const uint blockSize, const uint n) {
    const uint gid = get_global_id(0);
    if (gid >= n) {
        return;
    }

    const uint sumWithIdx = gid - blockSize;
    const bool flag = (sumWithIdx >= 0) && (sumWithIdx < n);
    // Хотел написать: (но не получалось скастить bool -> int)
    // sumWith = flag * as[sumWithIdx]
    const uint sumWith = flag ? as[sumWithIdx] : 0;
    result[gid] = as[gid] + sumWith;
} 

__kernel void radix_sort(const __global uint *as, __global uint *as_sorted_dst,
                         const __global uint *prefix_sums, const __global uint * cnts,
                         const uint shiftR, const uint n) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    const uint groupId = get_group_id(0);
    const uint wgNum = get_num_groups(0);

    if (gid >= n) {
        return;
    }

    const uint curVal = (as[gid] >> shiftR) & MASK;
    const uint curValAuxPos = wgNum * curVal + groupId;
    uint offset = prefix_sums[curValAuxPos] - cnts[curValAuxPos];

    // With local memory storage
    __local uint a[WORK_SIZE];
    a[lid] = as[gid];
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int i = 0; i < lid; ++i) {
        offset += curVal == ((a[i] >> shiftR) & MASK) ? 1 : 0;
    }

    // Without local memory
    // for (int i = groupId * WORK_SIZE; i < gid; ++i) {
    //     offset += curVal == ((as[i] >> shiftR) & MASK) ? 1 : 0;
    // }

    as_sorted_dst[offset] = as[gid];
}
