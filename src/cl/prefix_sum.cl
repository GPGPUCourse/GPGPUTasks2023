__kernel void prefixSum_upSweep(__global unsigned *as, int stepSize) {
    int i = get_global_id(0) * stepSize + stepSize - 1;
    as[i] += as[i - stepSize / 2];
}

__kernel void prefixSum_downSweep(__global unsigned *as, int blockSize) {
    int wi = get_global_id(0);
    if (!wi)
        return;
    int i = wi * blockSize + blockSize / 2 - 1;
    as[i] += as[i - blockSize / 2];
}
