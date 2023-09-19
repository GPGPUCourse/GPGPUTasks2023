typedef unsigned int uint_;
#define VALUES_PER_WORK_ITEM 64

__kernel void sum1(__global const uint_ *src, __global uint_ *res, const uint_ size) {
    const uint_ id = get_global_id(0);

    if (id >= size) {
        return;
    }

    atomic_add(res, src[id]);
}

__kernel void sum2(__global const uint_ *src, __global uint_ *res, const uint_ size) {
    const uint_ id = get_global_id(0);
    uint_ value = 0;
    for (uint_ i = 0; i < VALUES_PER_WORK_ITEM; i++) {
        uint_ indx = id * VALUES_PER_WORK_ITEM + i;

        if (indx < size) {
            value += src[indx];
        }
    }
    atomic_add(res, value);
}

__kernel void sum3(__global const uint_ *src, __global uint_ *res, const uint_ size) {
    uint_ gid = get_group_id(0);
    uint_ lid = get_local_id(0);
    uint_ wsize = get_local_size(0);
    uint_ value = 0;
    for (uint_ i = 0; i < VALUES_PER_WORK_ITEM; i++) {
        uint_ indx = gid * wsize * VALUES_PER_WORK_ITEM  +  wsize * i + lid;

        if (indx < size) {
            value += src[indx];
        }
    }
    atomic_add(res, value);
}