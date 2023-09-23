#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

__kernel void mandelbrot(__global float* results,
                         unsigned int width, unsigned int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         unsigned int iters) {
    // Почему трешхолд 256? На вики написано, что достаточно проверять для 2 (если модуль больше 2, то последовательность расходится)
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    int id_x = get_global_id(0);
    int id_y = get_global_id(1);

    if (id_x >= width || id_y >= height)
    {
        return;
    }

    float x = fromX + (id_x + 0.5f) * sizeX / width;
    float y = fromY + (id_y + 0.5f) * sizeY / height;

    float x_0 = x;
    float y_0 = y;
    int iteration = 0;
    for (; iteration < iters; iteration++) {
        if (x * x + y * y > threshold2) {
            break;
        }
        float buf_x = x;
        float buf_y = y;
        x = buf_x * buf_x - buf_y * buf_y + x_0;
        y = 2 * buf_x * buf_y + y_0;
    }

    float result = iteration;

    result = 1.0f * result / iters;
    results[id_y * width + id_x] = result;

    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
}