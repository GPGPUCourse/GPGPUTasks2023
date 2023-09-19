#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* results,
                   unsigned int width, unsigned int height,
                   float fromX, float fromY,
                   float sizeX, float sizeY,
                   unsigned int iters, int smoothing, int antiAliasing)
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    int id_x = get_global_id(0) % width, id_y = get_global_id(0) / width;

    if (id_x >= width || id_y >= height) {
        return;
    }

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;
    
    float x0, y0, x, y;

    float result, finalValue = 0.0f;
    float step = 1.0f / ((float)(antiAliasing) + 1.0f);
    int iter;

    for (int j = 0; j < antiAliasing; ++j) {
        for (int i = 0; i < antiAliasing; ++i) {
            x0 = fromX + (id_x + step * ((float)i + 1.0f)) * sizeX / width;
            y0 = fromY + (id_y + step * ((float)j + 1.0f)) * sizeY / height;
            x = x0, y = y0;
            iter = 0;
            for (; iter < iters; ++iter) {
                float xPrev = x;
                x = x * x - y * y + x0;
                y = 2.0f * xPrev * y + y0;
                if ((x * x + y * y) > threshold2) {
                    break;
                }
            }
            result = (float)iter;
            // if (smoothing && iter != iters) {
            //     result = result - logf(logf(sqrtf(x * x + y * y)) / logf(threshold)) / logf(2.0f);
            // }
            finalValue += 1.0f * result / (float)iters;
        }
    }
    results[id_y * width + id_x] = finalValue / (float)(antiAliasing * antiAliasing);
}
