#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define RESULT_INDEX(width, i, j) ((j * width) + i)

__kernel void mandelbrot(__global float *results, const unsigned int iters, const float fromX, const float fromY,
                         const float sizeX, const float sizeY, const int smoothing) {
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    const unsigned int width = get_global_size(0);
    const unsigned int height = get_global_size(1);
    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);


    float x0 = fromX + (i + 0.5f) * sizeX / width;
    float y0 = fromY + (j + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    int threshold_surpassed = 0;

    int iter = 0;
    int cnt = 0;
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        threshold_surpassed = threshold_surpassed || (x * x + y * y) > threshold2;
        cnt += (!threshold_surpassed);
    }
    float result = cnt;
    if (smoothing && iter != iters) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    result = 1.0f * result / iters;
    results[j * width + i] = result;
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
}
