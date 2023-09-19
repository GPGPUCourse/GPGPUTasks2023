#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *outBuf,
                         uint nx, uint ny,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         uint iterations, int smoothing)
{
    const float THRESHOLD = 256.0;
    const float THRESHOLD_SQR = THRESHOLD * THRESHOLD;

    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);
    if (ix >= nx || iy >= ny) {
        return;
    }
    size_t flatIdx = ix + nx * iy;

    float2 z0;
    // 2 * 4 = 8 ops
    z0.x = fromX + (ix + 0.5f) * sizeX / nx;
    z0.y = fromY + (iy + 0.5f) * sizeY / ny;
    float2 z = z0;

    float zSqr = 0.0f;
    uint iter = 0;
    // 10 ops / iter
    for (; iter < iterations; ++iter) {
        float2 z1;
        // 7 ops
        z1.x = z.x * z.x - z.y * z.y + z0.x;
        z1.y = 2.0f * z.x * z.y + z0.y;
        z = z1;

        // 3 ops
        zSqr = z.x * z.x + z.y * z.y;
        if (zSqr >= THRESHOLD_SQR) {
            break;
        }
    }

    float result = iter;
    if (smoothing && iter != iterations) {
        result = result - log(log(sqrt(zSqr)) / log(THRESHOLD)) / log(2.0f);
    }
    // 1 op (when measuring)
    result /= iterations;
    outBuf[flatIdx] = result;

    // Total: (9 + 10 * iter) ops

    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
}
