#ifdef __CLION_IDE__
#include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

__kernel void mandelbrot(__global float *out_buf,
                         uint nx, uint ny,
                         float from_x, float from_y,
                         float size_x, float size_y,
                         uint iterations, int smoothing)
{
    const float THRESHOLD = 256.0;
    const float THRESHOLD_SQR = THRESHOLD * THRESHOLD;

    size_t ix = get_global_id(0);
    size_t iy = get_global_id(1);
    if (ix >= nx || iy >= ny) {
        return;
    }
    size_t flat_idx = ix + nx * iy;

    float2 z0;
    // 2 * 4 = 8 ops
    z0.x = from_x + (ix + 0.5f) * size_x / nx;
    z0.y = from_y + (iy + 0.5f) * size_y / ny;
    float2 z = z0;

    float z_sqr = 0.0f;
    uint iter = 0;
    // 10 ops / iter
    for (; iter < iterations; ++iter) {
        float2 z1;
        // 7 ops
        z1.x = z.x * z.x - z.y * z.y + z0.x;
        z1.y = 2.0f * z.x * z.y + z0.y;
        z = z1;

        // 3 ops
        z_sqr = z.x * z.x + z.y * z.y;
        if (z_sqr >= THRESHOLD_SQR) {
            break;
        }
    }

    float result = iter;
    if (smoothing && iter != iterations) {
        result = result - log(log(sqrt(z_sqr)) / log(THRESHOLD)) / log(2.0f);
    }
    // 1 op (when measuring)
    result /= iterations;
    out_buf[flat_idx] = result;

    // Total: (9 + 10 * iter) ops

    // TODO если хочется избавиться от зернистости и дрожания при интерактивном погружении, добавьте anti-aliasing:
    // грубо говоря, при anti-aliasing уровня N вам нужно рассчитать не одно значение в центре пикселя, а N*N значений
    // в узлах регулярной решетки внутри пикселя, а затем посчитав среднее значение результатов - взять его за результат для всего пикселя
    // это увеличит число операций в N*N раз, поэтому при рассчетах гигаплопс антиальясинг должен быть выключен
}
