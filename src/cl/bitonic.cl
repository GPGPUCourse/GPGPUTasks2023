#ifdef __CLION_IDE__
    #include <libgpu/opencl/cl/clion_defines.cl>
#endif

#line 6

#define Swap(array, i, j) { float tmp = array[i]; array[i] = array[j]; array[j] = tmp; }
#define CompareAndSwap(array, i, j) if (array[i] > array[j]) Swap(array, i, j)

__kernel void bitonic(__global float *as,
                      const unsigned int len,
                      const unsigned int layering)
{
    unsigned int id = get_global_id(0);
    unsigned int block = id / layering;
    unsigned int i = 2 * block * layering + (id % layering);
    unsigned int j = i + layering;
    int direction = id / (len / 2) % 2;
    if (direction == 0)
    {
        CompareAndSwap(as, i, j);
    }
    else
    {
        CompareAndSwap(as, j, i);
    }
}
