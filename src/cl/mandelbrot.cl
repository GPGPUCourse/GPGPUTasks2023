#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *results,
                         unsigned int width, unsigned int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         unsigned int iters, const unsigned int smoothing) {
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    const unsigned int index_y = get_global_id(1);
    if (index_y >= height)
        return;

    const unsigned int index_x = get_global_id(0);

    float x0 = fromX + (index_x + 0.5f) * sizeX / width;
    float y0 = fromY + (index_y + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }
    float result = iter;

    if (smoothing && iter != iters) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    result = 1.0f * result / iters;
    results[index_y * width + index_x] = result;
}
