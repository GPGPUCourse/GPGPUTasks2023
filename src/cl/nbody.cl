#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GRAVITATIONAL_FORCE 0.0001

/// Calculate accelerations for everything.
///
/// \param pxs, pys Object positions
/// \param vxs, vys Object velocities
/// \param mxs Object masses
/// \param[out] dvx2d, dvy2d Object accelerations
/// \param N Number of objects
/// \param t Time
__kernel void nbody_calculate_force_global(
    __global const float *pxs,
    __global const float *pys,
    __global const float *vxs,
    __global const float *vys,
    __global const float *mxs,
    __global float *dvx2d,
    __global float *dvy2d,
    int N,
    int t
) {
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float *dvx = dvx2d + t * N;
    __global float *dvy = dvy2d + t * N;

    // This object i
    float x0 = pxs[i];
    float y0 = pys[i];
    float m0 = mxs[i];

    for (int j = 0; j < N; ++j) {
        if (j == i)
            continue;
        float x1 = pxs[j], y1 = pys[j], m1 = mxs[j];
        float dx = x1 - x0;
        float dy = y1 - y0;
        float dr2 = fmax(100.f, dx * dx + dy * dy);
        float dr2_inv = 1.f / dr2;
        float dr_inv = sqrt(dr2_inv);

        float ex = dx * dr_inv;
        float ey = dy * dr_inv;

        float fx = ex * dr2_inv * GRAVITATIONAL_FORCE;
        float fy = ey * dr2_inv * GRAVITATIONAL_FORCE;

        dvx[i] += m1 * fx;
        dvy[i] += m1 * fy;
    }
}

/// Apply calculated accelerations to the velocities,
/// then apply the velocities.
///
/// \param pxs, pys Object positions
/// \param vxs, vys Object velocities
/// \param mxs Object masses
/// \param dvx2d, dvy2d Object accelerations
/// \param N Number of objects
/// \param t Time
__kernel void nbody_integrate(
        __global float *pxs,
        __global float *pys,
        __global float *vxs,
        __global float *vys,
        __global const float *mxs,
        __global const float *dvx2d,
        __global const float *dvy2d,
        int N,
        int t
) {
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float * dvx = dvx2d + t * N;
    __global float * dvy = dvy2d + t * N;

    vxs[i] += dvx[i];
    vys[i] += dvy[i];
    pxs[i] += vxs[i];
    pys[i] += vys[i];
}
