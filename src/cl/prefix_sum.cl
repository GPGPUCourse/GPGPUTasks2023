__kernel void prefix_sum(__global uint *as, __global uint *result, const uint blockSize, const uint n) {
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