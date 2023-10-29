__kernel void bitonic(__global float *as, const uint superBlockSize, const uint smallBlockSize, const uint n) {
    const uint gid = get_global_id(0);
    if (gid >= n) {
        return;
    }

    uint start = (gid / smallBlockSize) * (smallBlockSize * 2);
    uint offset = gid % smallBlockSize;

    bool isAscending = (gid / superBlockSize) % 2 == 0;
    uint i = start + offset + (isAscending ? smallBlockSize : 0);
    uint j = start + offset + (isAscending ? 0 : smallBlockSize);

    if (i < n && j < n && as[i] < as[j]) {
        float temp = as[i];
        as[i] = as[j];
        as[j] = temp;
    }
}
