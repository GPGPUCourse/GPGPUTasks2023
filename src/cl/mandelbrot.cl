#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__constant const float threshold = 256.0f;
__constant const float threshold2 = threshold * threshold;

__kernel void mandelbrot(
    __global float *results,
    unsigned width,
    unsigned height,
    float fromX,
    float fromY,
    float sizeX,
    float sizeY,
    unsigned iters,
    /* bool */ int smoothing 
)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
    float x0 = fromX + (i + 0.5f) * sizeX / width;
    float y0 = fromY + (j + 0.5f) * sizeY / height;
    float x = x0, y = y0;
    int iter;
    for (iter = 0; iter < iters; ++iter) {
        float new_x = x * x - y * y + x0;
        float new_y = 2 * x * y + y0;
        x = new_x;
        y = new_y;
        if (x * x + y * y > threshold2)
            break;
    }
    float result = iter;
    if (smoothing && iter != iters) {
        result = result - log(log(sqrt(x * x + y * y))) / log(threshold) / log(2.0f);
    }
    results[j * width + i] = result / iters;
}
