#ifdef __CLION_IDE__

#include <libgpu/opencl/cl/clion_defines.cl>

#endif

#line 6

#define WAVE_SIZE 32
__kernel void mandelbrot(__global float *out,
                         unsigned int width, unsigned int height,
                         float fromX, float fromY,
                         float sizeX, float sizeY,
                         unsigned int iterationsLimit, unsigned int smoothing) {

    const float threshold = 256.0f * 256.0f;

    const unsigned thread_id = get_local_id(0);
    const unsigned wg_size = get_local_size(0);
    const unsigned cu_id = get_group_id(0);
    const unsigned cu_cnt = get_num_groups(0);

    // WORKS_SIZE per CU

    const unsigned work_size = (width * height + cu_cnt - 1) / cu_cnt;
    const unsigned cu_offset = work_size * cu_id;


    unsigned thread_offset = cu_offset + thread_id;
    // start value for (x, y)


    const unsigned end_offset = min(width * height, cu_offset + work_size);

    while (thread_offset < end_offset) {

        float i, j;
        float x0, y0, x, y, it;


        i = (thread_offset) % width;
        j = (thread_offset) / width;

        x0 = fromX + (i + 0.5f) * sizeX / width;
        y0 = fromY + (j + 0.5f) * sizeY / height;

        x = x0;
        y = y0;
        it = iterationsLimit;

        float result;
        int iter = 0;
        for (; iter < iterationsLimit; ++iter) {


            float xPrev = x;

            x = fma(y, -y, fma(x, x, x0));
            y = fma(2.0f * xPrev, y, y0);

            if ((fma(x, x, y * y)) > threshold) {
                it += iter;
                // hack to avoid creating instructions to change exec mask
                x = 0;
                y = 0;
                x0 = 0;
                y0 = 0;
            }

        }

        if(it - iterationsLimit > 1e-6) {
            it = it - iterationsLimit;
        }

        result = it / iterationsLimit;

        if (smoothing && (it - iterationsLimit) > 1e-6) {
            result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
        }

        out[thread_offset] = result;
        thread_offset += wg_size;

    }

}

__kernel void mandelbrot_naive(__global float *out, const unsigned int width, const unsigned int height,
                               const float fromX, const float fromY, const float sizeX, const float sizeY,
                               const unsigned int iterationsLimit, const int smoothing) {
    const float threshold = 256.0f * 256.0f;

    const unsigned int i = get_global_id(0);
    const unsigned int j = get_global_id(1);
    float x0 = fromX + (i + 0.5f) * sizeX / width;
    float y0 = fromY + (j + 0.5f) * sizeY / height;

    float x = x0;
    float y = y0;

    int iter = 0;
    for (; iter < iterationsLimit; ++iter) {
        float xPrev = x;
        x = x * x - y * y + x0;
        y = 2.0f * xPrev * y + y0;
        if ((x * x + y * y) > threshold) {
            break;
        }
    }
    float result = iter;
    if (smoothing && iter != iterationsLimit) {
        result = result - log(log(sqrt(x * x + y * y)) / log(threshold)) / log(2.0f);
    }

    result = 1.0f * result / iterationsLimit;
    out[j * width + i] = result;
}