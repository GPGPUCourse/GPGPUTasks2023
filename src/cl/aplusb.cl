#ifdef __CLION_IDE__
// Этот include виден только для CLion парсера, это позволяет IDE "знать" ключевые слова вроде __kernel, __global
// а также уметь подсказывать OpenCL методы, описанные в данном инклюде (такие как get_global_id(...) и get_local_id(...))
#include "clion_defines.cl"

#endif

#line 8// Седьмая строчка теперь восьмая (при ошибках компиляции в логе компиляции будут указаны корректные строчки благодаря этой директиве)

// TODO 5 реализуйте кернел:
// - От обычной функции кернел отличается модификатором __kernel и тем, что возвращаемый тип всегда void
// - На вход дано три массива float чисел; единственное, чем они отличаются от обычных указателей - модификатором __global, т.к. это глобальная память устройства (видеопамять)
// - Четвертым и последним аргументом должно быть передано количество элементов в каждом массиве (unsigned int, главное, чтобы тип был согласован с типом в соответствующем clSetKernelArg в T0D0 10)

__kernel void aplusb(__global float *a, __global float *b, __global float *c, const unsigned int n) {

    size_t index = get_global_id(0);
    if (index < n) {
        c[index] = a[index] + b[index];
    }

}

#define WAVE_SIZE   32
__kernel void vecadd_pt(__global float *a, __global float *b, __global float *c, const unsigned int n) {
    size_t laneId = get_global_id(0) % WAVE_SIZE;
    size_t wave_id = get_global_id(0) / WAVE_SIZE;
    size_t wave_offset_load = WAVE_SIZE * wave_id + laneId;
    size_t wave_offset_store = wave_offset_load;

    float aa[2];
    float bb[2];
    if (wave_offset_load < n) {
        aa[0] = a[wave_offset_load];
        bb[0] = b[wave_offset_load];
        wave_offset_load += get_global_size(0);
    }

    unsigned i = 1;
    while (wave_offset_store < n) {
        if (wave_offset_load < n) {
            aa[i % 2] = a[wave_offset_load];
            bb[i % 2] = b[wave_offset_load];
            wave_offset_load += get_global_size(0);
        }
        c[wave_offset_store] = aa[(i + 1) % 2] + bb[(i + 1) % 2];
        wave_offset_store += get_global_size(0);
        i++;
    }
}