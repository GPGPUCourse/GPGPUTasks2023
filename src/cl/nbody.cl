#ifdef __CLION_IDE__
#include "clion_defines.cl"
#endif

#line 6

#define GRAVITATIONAL_FORCE 0.0001

// Quake III fast inverse square root
float q_rsqrt(float number)
{
	long i;
	float x2, y;
	const float threehalfs = 1.5F;

	x2 = number * 0.5F;
	y  = number;
	i  = * ( long * ) &y;                       // evil floating point bit level hacking
	i  = 0x5f3759df - ( i >> 1 );               // what the fuck?
	y  = * ( float * ) &i;
	y  = y * ( threehalfs - ( x2 * y * y ) );   // 1st iteration
//	y  = y * ( threehalfs - ( x2 * y * y ) );   // 2nd iteration, this can be removed

	return y;
}

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
//    float m0 = mxs[i];

	for (int j = 0; j < N; ++j) {
		if (j == i)
			continue;

		float x1 = pxs[j];
		float y1 = pys[j];
		float m1 = mxs[j];

		float dx = x1 - x0;
		float dy = y1 - y0;
		float dr2 = max(100.f, dx * dx + dy * dy);

		float dr_inv = q_rsqrt(dr2);
		float dr2_inv = dr_inv * dr_inv;

		float ex = dx * dr_inv;
		float ey = dy * dr_inv;

		float fx = ex * dr2_inv * GRAVITATIONAL_FORCE;
		float fy = ey * dr2_inv * GRAVITATIONAL_FORCE;

		dvx[i] += m1 * fx;
		dvy[i] += m1 * fy;
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
