#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define GRAVITATIONAL_FORCE 0.0001

__kernel void nbody_calculate_force_global(
    __global float * pxs, __global float * pys,
    __global float *vxs, __global float *vys,
    __global const float *mxs,
    __global float * dvx2d, __global float * dvy2d,
    int N,
    int t)
{
    unsigned int i = get_global_id(0);

    if (i >= N)
        return;

    __global float * dvx = dvx2d + t * N;
    __global float * dvy = dvy2d + t * N;

    float x0 = pxs[i];
    float y0 = pys[i];
    float m0 = mxs[i];

    for (int j = 0; j < N; j++) {
        if (j == i) {
            continue;
        }

        float x1 = pxs[j];
        float y1 = pys[j];
        float m1 = mxs[j];

        float dist_x = x1 - x0;
        float dist_y = y1 - y0;

        float dist = sqrt(max(100.f, dist_x * dist_x + dist_y * dist_y));

        float force_x = dist_x * GRAVITATIONAL_FORCE / (dist * dist * dist);
        float force_y = dist_y * GRAVITATIONAL_FORCE / (dist * dist * dist);

        dvx[i] += m1 * force_x;
        dvy[i] += m1 * force_y;
    }
}

__kernel void nbody_integrate(
        __global float * pxs, __global float * pys,
        __global float *vxs, __global float *vys,
        __global const float *mxs,
        __global float * dvx2d, __global float * dvy2d,
        int N,
        int t)
{
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
