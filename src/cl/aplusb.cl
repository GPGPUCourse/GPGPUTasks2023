__kernel void aplusb(__global float* as, __global float* bs, __global float* cs, unsigned int n) {
    size_t global_id = get_global_id(0);

    if (global_id >= n) {
        return;
    }

    cs[global_id] = as[global_id] + bs[global_id];
}
