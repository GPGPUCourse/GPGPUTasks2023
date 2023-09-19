#include "CL/cl.h"
#include "CL/cl_platform.h"
#include <cmath>
#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global cl_float *results, const cl_uint width, const cl_uint height, const cl_float fromX,
                         const cl_float fromY, const cl_float sizeX, const cl_float sizeY, const cl_uint iters,
                         const cl_bool smoothing) {
    const cl_float threshold = 256.0f;
    const cl_float threshold2 = threshold * threshold;

    cl_uint id = get_global_id(0);

    cl_uint i = id % width;
    cl_uint j = id / width;

    if (!(i < width && j < height)) return;

    cl_float x0 = fromX + (i + 0.5f) * sizeX / width;
    cl_float y0 = fromY + (j + 0.5f) * sizeY / height;

    cl_float x = x0;
    cl_float y = y0;

    cl_int iter = 0;
    for (; iter < iters; ++iter) {
        cl_float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold2) {
            break;
        }
    }

    cl_float result = iter;
    if (smoothing && iter != iters) {
        result = result - logf(logf(sqrtf(x * x + y * y)) / logf(threshold)) / logf(2.0f);
    }

    result = 1.0f * result / iters;
    results[j * width + i] = result;
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
}
