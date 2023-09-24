// vim: syntax=c

#ifdef __CLION_IDE__
    // Этот include виден только для CLion парсера,
    // это позволяет IDE "знать" ключевые слова вроде __kernel, __global
    // а также уметь подсказывать OpenCL методы,
    // описанные в данном инклюде (такие как get_global_id(...) и get_local_id(...))
    #include "clion_defines.cl"
#endif

#line 12

__kernel void aplusb(
    __global const float * as,
    __global const float * bs,
    __global float * cs,
    unsigned int n
) {
    const unsigned int i = get_global_id(0);

    if (i >= n) {
        return;
    }

    cs[i] = as[i] + bs[i];
}
