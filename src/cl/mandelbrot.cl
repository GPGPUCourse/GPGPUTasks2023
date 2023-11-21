#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* res, const uint w, const uint h,
                         const float fromX, const float fromY,
                         const float sizeX, const float sizeY,
                         const uint iters, const uint smoothing) {
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    const uint i = get_global_id(0);
    const uint j = get_global_id(1);

    if (i >= w || j >= h) {
        return;
    }

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    float x0 = fromX + (i + 0.5f) * sizeX / w;
    float y0 = fromY + (j + 0.5f) * sizeY / h;

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
    res[j * w + i] = result;
}
