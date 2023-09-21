#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

float mandelbrot_at_point(float x0, float y0, float iters, const float threshold) {
    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold) {
            break;
        }
    }
    return (float)iter / iters;
}

__kernel void mandelbrot(__global float * results,
                         unsigned int width, unsigned int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         unsigned int iters, unsigned int levelAliasing)
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    int id = get_global_id(0);
    int idx = id % width, idy = id / width;

    if (idy >= height) {
        return;
    }

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    float res = 0;
    for (int i = 0; i < levelAliasing; i++) {
        for (int j = 0; j < levelAliasing; j++) {
            float x0 = fromX + (idx + (1.0f + i) / (1.0f + levelAliasing)) * sizeX / width;
            float y0 = fromY + (idy + (1.0f + j) / (1.0f + levelAliasing)) * sizeY / height;
            res += mandelbrot_at_point(x0, y0, iters, threshold2);
        }
    }
    results[id] = res / (levelAliasing * levelAliasing);
}
