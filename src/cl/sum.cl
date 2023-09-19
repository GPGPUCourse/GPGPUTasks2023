#ifdef __CLION_IDE__
    #include "clion_defines.cl"
#endif

#ifndef WORKGROUP_SIZE
    #error "Define workgroup size"
#endif

#ifndef VALUES_PER_WORKITEM
    #error "Define values per work item"
#endif


__kernel void sum1_atomic(__global const uint *inArr, __global uint *outSum, ulong n) {
    const size_t gid = get_global_id(0);
    if (gid >= n) {
        return;
    }
    atomic_add(outSum, inArr[gid]);
}


__kernel void sum2_loop(__global const uint *inArr, __global uint *outSum, ulong n) {
    const size_t gid = get_global_id(0);
    uint sumAtom = 0;
    for (size_t item = 0; item < VALUES_PER_WORKITEM; ++item) {
        size_t idx = gid * VALUES_PER_WORKITEM + item;
        if (idx >= n) {
            break;
        }
        sumAtom += inArr[idx];
    }
    atomic_add(outSum, sumAtom);
}


__kernel void sum3_loop_coalesced(__global const uint *inArr, __global uint *outSum, ulong n) {
    const size_t lid = get_local_id(0);
    const size_t wid = get_group_id(0);
    const size_t wgsz = get_local_size(0);
    uint sumAtom = 0;
    for (size_t item = 0; item < VALUES_PER_WORKITEM; ++item) {
        size_t idx = (wid * VALUES_PER_WORKITEM + item) * wgsz + lid;
        if (idx >= n) {
            break;
        }
        sumAtom += inArr[idx];
    }
    atomic_add(outSum, sumAtom);
}


__kernel void sum4_local_mem(__global const uint *inArr, __global uint *outSum, ulong n) {
    __local uint buf[WORKGROUP_SIZE];
    const size_t gid = get_global_id(0);
    const size_t lid = get_local_id(0);

    buf[lid] = gid < n ? inArr[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid != 0) {
        return;
    }

    uint sumAtom = 0;
    for (size_t i = 0; i < WORKGROUP_SIZE; ++i) {
        sumAtom += buf[i];
    }

    atomic_add(outSum, sumAtom);
}

void eval_local_sum(__global const uint *inArr, __local uint *outBuf, size_t n) {
    const size_t gid = get_global_id(0);
    const size_t lid = get_local_id(0);

    outBuf[lid] = gid < n ? inArr[gid] : 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (size_t nVals = WORKGROUP_SIZE; (nVals /= 2) > 0;) {
        if (lid < nVals) {
            outBuf[lid] += outBuf[lid + nVals];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}


__kernel void sum5_tree_local(__global const uint *inArr, __global uint *outSum, ulong n) {
    __local uint buf[WORKGROUP_SIZE];
    const size_t lid = get_local_id(0);
    eval_local_sum(inArr, buf, n);
    if (lid == 0) {
        atomic_add(outSum, buf[0]);
    }
}


__kernel void sum6_tree_global(__global const uint *inArr, __global uint *outArr, ulong n) {
    __local uint buf[WORKGROUP_SIZE];
    const size_t lid = get_local_id(0);
    const size_t wid = get_group_id(0);
    eval_local_sum(inArr, buf, n);
    if (lid == 0) {
        outArr[wid] = buf[0];
    }
}
