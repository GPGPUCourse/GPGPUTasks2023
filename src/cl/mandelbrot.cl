#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

float mandelbrot_helper(float x0, float y0, float max_iterations, const float infinity) {
    float x = x0;
    float y = y0;

    int iteration = 0;
    for (; iteration < max_iterations; ++iteration) {
        float x_new = x * x - y * y + x0;
        y = 2.0f * x * y + y0;
        x = x_new;
        if ((x * x + y * y) > infinity) {
            break;
        }
    }
    return (float)iteration / max_iterations;
}

__kernel void mandelbrot(__global float * result_buf,
                         unsigned int width, unsigned int height,
                         float start_x, float start_y,
                         float size_x, float size_y,
                         unsigned int max_iterations, unsigned int anti_aliasing_level)
{
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    int id = get_global_id(0);
    int idx = id % width;
    int idy = id / width;
    float sum = 0;

    if (idx >= width || idy >= height)
        return;

    for (int i = 0; i < anti_aliasing_level; i++)
        for (int j = 0; j < anti_aliasing_level; j++) {
            float x0 = start_x + (idx +  (i + 1.0f) / (anti_aliasing_level + 1.0f)) * size_x / width;
            float y0 = start_y + (idy +  (j + 1.0f) / (anti_aliasing_level + 1.0f)) * size_y / height;
            sum += mandelbrot_helper(x0, y0, max_iterations, 256.0f * 256.0f);
        }

    result_buf[id] = sum / (anti_aliasing_level * anti_aliasing_level);
}
