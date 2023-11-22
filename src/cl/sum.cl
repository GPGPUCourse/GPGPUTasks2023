#define WG_SIZE 128

__kernel void sum_atomic(__global uint* as, __global uint* res, const uint n) {
    const uint gid = get_global_id(0);
    if (gid >= n) {
        return;
    }

    atomic_add(&res[0], as[gid]);
}

__kernel void sum_loop(__global uint* as, __global uint* res, const uint n) {
    // Можно было вынести в аргументы функции, но хотелось, чтобы все функции были одинаково параметризованы
    const uint numItems = 32;
    const uint gid = get_global_id(0);
    if (gid >= n) {
        return;
    }

    uint local_res = 0;
    for (int i = 0; i < numItems; i++) {
        uint idx = gid * numItems + i;
        if (idx < n) {
            local_res += as[idx];
        }
    }

    atomic_add(&res[0], local_res);
}

__kernel void sum_loop_coalesced(__global uint* as, __global uint* res, const uint n) {
    const uint numIters = 32;
    const uint lid = get_local_id(0);
    const uint wgId = get_group_id(0);

    uint local_res = 0;
    for (int i = 0; i < numIters; ++i) {
        uint idx = wgId * WG_SIZE * numIters + i * WG_SIZE + lid;
        if (idx < n) {
            local_res += as[idx];
        }
    }
    
    atomic_add(&res[0], local_res);
}


__kernel void sum_local(__global uint* as, __global uint* res, const uint n) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    if (gid >= n) {
        return;
    }

    __local uint a[WG_SIZE];
    if (lid < WG_SIZE) {
        a[lid] = as[gid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        uint local_res = 0;
        for (int i = 0; i < WG_SIZE; ++i) {
            local_res += a[i];
        }

        atomic_add(&res[0], local_res);
    }
}


__kernel void sum_tree(__global uint* as, __global uint* res, const uint n) {
    const uint gid = get_global_id(0);
    const uint lid = get_local_id(0);
    if (gid >= n) {
        return;
    }

    __local uint buf[WG_SIZE];
    if (lid < WG_SIZE) {
        buf[lid] = as[gid];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int nVals = WG_SIZE; nVals > 1; nVals /= 2) {
        if (2 * lid < nVals) {
            buf[lid] += buf[lid + nVals / 2];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }


    if (lid == 0) {
        atomic_add(&res[0], buf[0]);
    }
}