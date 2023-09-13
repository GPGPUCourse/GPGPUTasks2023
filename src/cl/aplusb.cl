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

__kernel void aplusb(__global float* a, __gloabl float* b, __global float* c, unsigned int n) {
    // Узнать, какой workItem выполняется в этом потоке поможет функция get_global_id
    // см. в документации https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // OpenCL Compiler -> Built-in Functions -> Work-Item Functions
    size_t totalWorkGroups = get_num_groups(0);
    size_t workGroupSize, workItemSize;
    if (n % totalWorkGroups == 0) {
        workGroupSize = (size_t)n / totalWorkGroups;
    } else {
        workGroupSize = (size_t)n / totalWorkGroups + 1;
    }
    size_t workGroup = get_group_id(0), workItem = get_local_id(0);
    size_t localWorkItems = get_local_size(0);
    if (workGroupSize % localWorkItems == 0) {
        workItemSize = workGroupSize / localWorkItems;
    } else {
        workItemSize = workGroupSize / localWorkItems + 1;
    }

    for (int i = workGroupSize * workGroup + workItemSize * workItem; i < workGroupSize * workGroup + workItemSize * (workItem + 1); i++) {
        if (i >= n) {
            return;
        }
        c[i] = a[i] + b[i];
    }

    // P.S. В общем случае количество элементов для сложения может быть некратно размеру WorkGroup, тогда размер рабочего пространства округлен вверх от числа элементов до кратности на размер WorkGroup
    // и в таком случае, если сделать обращение к массиву просто по индексу=get_global_id(0), будет undefined behaviour (вплоть до повисания ОС)
    // поэтому нужно либо дополнить массив данных длиной до кратности размеру рабочей группы,
    // либо сделать return в кернеле до обращения к данным в тех WorkItems, где get_global_id(0) выходит за границы данных (явной проверкой)
}
