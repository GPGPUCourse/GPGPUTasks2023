#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GRAVITATIONAL_FORCE 0.0001

__kernel void nbody_calculate_force_global(__global float *pxs, __global float *pys, __global float *vxs,
                                           __global float *vys, __global const float *mxs, __global float *dvx2d,
                                           __global float *dvy2d, int N, int t) {
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float *dvx = dvx2d + t * N;
    __global float *dvy = dvy2d + t * N;

    float x0 = pxs[i];
    float y0 = pys[i];
    float m0 = mxs[i];


    float dvx_acc = 0;
    float dvy_acc = 0;
    for (int j = 0; j < N; ++j) {
        if (j == i) {
            continue;
        }

        float x1 = pxs[j];
        float y1 = pys[j];
        float m1 = mxs[j];

        float dx = x1 - x0;
        float dy = y1 - y0;
        float dr2 = max(100.f, dx * dx + dy * dy);

        float dr2_inv = 1.f / dr2;
        float dr_inv = sqrt(dr2_inv);

        float ex = dx * dr_inv;
        float ey = dy * dr_inv;

        float fx = ex * dr2_inv * GRAVITATIONAL_FORCE;
        float fy = ey * dr2_inv * GRAVITATIONAL_FORCE;

        dvx_acc += m1 * fx;
        dvy_acc += m1 * fy;
    }
    dvx[i] += dvx_acc;
    dvy[i] += dvy_acc;
}

__kernel void nbody_integrate(__global float *pxs, __global float *pys, __global float *vxs, __global float *vys,
                              __global const float *mxs, __global float *dvx2d, __global float *dvy2d, int N, int t) {
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float *dvx = dvx2d + t * N;
    __global float *dvy = dvy2d + t * N;

    vxs[i] += dvx[i];
    vys[i] += dvy[i];
    pxs[i] += vxs[i];
    pys[i] += vys[i];
}
