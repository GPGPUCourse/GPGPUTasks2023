#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *img, 
                         const uint width, const uint height, 
                         const float fromX, const float fromY,
                         const float sizeX, const float sizeY, 
                         const uint iters, const unsigned smoothing) {
    const size_t index_x = get_global_id(0);
    const size_t index_y = get_global_id(1);

    if (index_x >= width || index_y >= height) {
        return;
    }

    // Same as CPU
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;
    // 8 ops
    float x0 = fromX + (index_x + 0.5f) * sizeX / width;
    float y0 = fromY + (index_y + 0.5f) * sizeY / height;
    float x = x0;
    float y = y0;

    int iter = 0;
    // 10 ops
    for (; iter < iters; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) >= threshold2) {
            break;
        }
    }
    float result = iter;
    if (smoothing && iter != iters) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }
    // 1 ops
    result /= iters;
    img[index_y * width + index_x] = result;

    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
}
