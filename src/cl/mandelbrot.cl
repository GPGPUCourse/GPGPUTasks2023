#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

kernel void mandelbrot(global float* outputResults, const unsigned int imageWidth, const unsigned int imageHeight, const float start_x, const float start_y, const float sizeWidth, const float sizeHeight, const unsigned int maxIter, const int isSmoothingEnabled) {
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
    const float escapeRadius = 256.0f;
    const float escapeRadiusSquared = escapeRadius * escapeRadius;

    int pixel_x = get_global_id(0);
    int pixel_y = get_global_id(1);

    if (pixel_x >= imageWidth || pixel_y >= imageHeight) {
        return;
    }

    float mandelbrot_x = start_x + (pixel_x + 0.5f) * sizeWidth / imageWidth;
    float mandelbrot_y = start_y + (pixel_y + 0.5f) * sizeHeight / imageHeight;

    float x = mandelbrot_x;
    float y = mandelbrot_y;

    int iter= 0;
    for (; iter < maxIter; ++iter) {
        const float previous_x = x;
        x = x * x - y * y + mandelbrot_x;
        y = 2.0f * previous_x * y + mandelbrot_y;
        if ((x * x + y * y) > escapeRadiusSquared) {
            break;
        }
    }

    float result = iter;
    if (isSmoothingEnabled && iter != maxIter) {
        result = result - log(log(sqrt(x * x + y * y)) / log(escapeRadius)) / log(2.0f);
    }

    result = 1.0f * result / maxIter;
    outputResults[pixel_y * imageWidth + pixel_x] = result;
}
