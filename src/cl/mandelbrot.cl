#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

int getIter(int iters, float *x, float *y, float x0, float y0, const float threshold2) {
    int iter = 0;
    for (; iter < iters; ++iter) {
        float xPrev = *x;
        *x = *x * *x - *y * *y + x0;
        *y = 2.0f * xPrev * *y + y0;
        if ((*x * *x + *y * *y) > threshold2) {
            break;
        }
    }
    return iter;
}

__kernel void mandelbrot(__global float *results, const unsigned int width, const unsigned int height,
                         const float fromX, const float fromY, const float sizeX, const float sizeY,
                         const unsigned int iters, int smoothing) {
    const float threshold = 256.0f;
    const float threshold2 = threshold * threshold;

    int anti_aliasing = 1;
    if (anti_aliasing < 1){
        anti_aliasing = 1;
    }
    int ix = get_global_id(0);
    int iy = get_global_id(1);

    if (ix < width && iy < height) {
        int iter = 0;
        float x = 0;
        float y = 0;
        for (int j = 0; j < anti_aliasing; ++j) {
            for (int i = 0; i < anti_aliasing; ++i) {
                float x0 = fromX + (ix + (anti_aliasing - 1 ? (float) i / (anti_aliasing - 1) : 0.5f)) * sizeX / width;
                float y0 = fromY + (iy + (anti_aliasing - 1 ? (float) j / (anti_aliasing - 1) : 0.5f)) * sizeY / height;
                x = x0;
                y = y0;

                iter += getIter(iters, &x, &y, x0, y0, threshold2);
            }
        }
        float result = anti_aliasing ? iter / (anti_aliasing * anti_aliasing) : iter;
        if (smoothing && iter != iters) {
            result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
        }

        result = 1.0f * result / iters;
        results[iy * width + ix] = result;
    }
    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
}
