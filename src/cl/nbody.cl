#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GRAVITATIONAL_FORCE 0.0001

__kernel void nbody_calculate_force_global(__global float *pxs,
                                           __global float *pys,
                                           __global float *vxs,
                                           __global float *vys,
                                           __global const float *mxs,
                                           __global float *dvx2d,
                                           __global float *dvy2d,
                                           int N,
                                           int t) {
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float *dvx = dvx2d + t * N;
    __global float *dvy = dvy2d + t * N;

    float x0 = pxs[i];
    float y0 = pys[i];
    float m0 = mxs[i];

    // TODO
}

__kernel void nbody_integrate(__global float *pxs,
                              __global float *pys,
                              __global float *vxs,
                              __global float *vys,
                              __global const float *mxs,
                              __global float *dvx2d,
                              __global float *dvy2d,
                              int N,
                              int t) {
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
