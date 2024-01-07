#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float* res,
                         const unsigned int width, const unsigned int height,
                         const float fromX, const float fromY,
                         const float sizeX, const float sizeY,
                         const unsigned int iters, const int smoothing)
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен

    int id_x = get_global_id(0) % width, id_y = get_global_id(0) / width;

    if (id_x >= width || id_y >= height)
        return;

    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    float x0 = fromX + (id_x + 0.5f) * sizeX;
    float y0 = fromY + (id_y + 0.5f) * sizeY;
    
    float x = x0, y = y0;

    int iter = 0;
    for (; iter < iters; ++iter) {
        float new_x = x * x - y * y + x0;
        float new_y = 2.0f * x * y + y0;
        x = new_x;
        y = new_y;
        if (x * x + y * y > threshold2)
            break;
    }

    float result = iter;
    
    if (smoothing && iter != iters) {
        result = result - log(log(sqrt(x * x + y * y))) / log(threshold) / log(2.0f);
    }
    
    res[id_y * width + id_x] = 1.0f * result / iters;
}