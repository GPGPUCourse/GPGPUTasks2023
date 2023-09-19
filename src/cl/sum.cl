#ifndef WORKGROUP_SIZE
#error "Define workgroup size"
#endif

#ifndef VALUES_PER_WORKITEM
#error "Define values per work item"
#endif

__kernel void sum1_atomic(__global const uint *inBuf, __global uint *outBuf, ulong n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    atomic_add(outBuf, inBuf[gid]);
}

__kernel void sum2_loop(__global const uint *inBuf, __global uint *outBuf, ulong n) {
    const size_t gid = get_global_id(0);
    uint sumAtom = 0;
    for (size_t item = 0; item < VALUES_PER_WORKITEM; ++item) {
        size_t idx = gid * VALUES_PER_WORKITEM + item;
        if (idx >= n) {
            break;
        }
        sumAtom += inBuf[idx];
    }
    atomic_add(outBuf, sumAtom);
}
