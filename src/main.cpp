#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <vector>

#include "error_handler.h"
#include "helpers.h"

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init()) {
        throw std::runtime_error("Can't init OpenCL driver!");
    }

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)

    cl_device_id selectedDevice = helpers::selectDevice();
    helpers::prettyPrintSelectedDevice(selectedDevice);

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)

    cl_int errcode_ret = CL_SUCCESS;

    helpers::Holder<cl_context, int (*)(_cl_context *)>
            contextHolder(clCreateContext(nullptr, 1, &selectedDevice, nullptr, nullptr, &errcode_ret),
                          clReleaseContext);

    if (errcode_ret != CL_SUCCESS) {
        eh::OCL_SAFE_CALL(errcode_ret);
    }

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)

    helpers::Holder<cl_command_queue, int (*)(_cl_command_queue *)>
            queueHolder(clCreateCommandQueue(contextHolder.get(), selectedDevice, 0, &errcode_ret),
                        clReleaseCommandQueue);

    if (errcode_ret != CL_SUCCESS) {
        eh::OCL_SAFE_CALL(errcode_ret);
    }

    unsigned int n = 100 * 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;
    std::cout << std::endl;

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)

    const unsigned int sizeBuff = sizeof(float) * n;

    helpers::Holder<cl_mem, int (*)(_cl_mem *)>
            memAsHolder(clCreateBuffer(contextHolder.get(), CL_MEM_READ_ONLY, sizeBuff, nullptr, &errcode_ret),
                        clReleaseMemObject);
    eh::OCL_SAFE_CALL(errcode_ret);

    helpers::Holder<cl_mem, int (*)(_cl_mem *)>
            memBsHolder(clCreateBuffer(contextHolder.get(), CL_MEM_READ_ONLY, sizeBuff, nullptr, &errcode_ret),
                        clReleaseMemObject);
    eh::OCL_SAFE_CALL(errcode_ret);

    helpers::Holder<cl_mem, int (*)(_cl_mem *)>
            memCsHolder(clCreateBuffer(contextHolder.get(), CL_MEM_WRITE_ONLY, sizeBuff, nullptr, &errcode_ret),
                        clReleaseMemObject);
    eh::OCL_SAFE_CALL(errcode_ret);

    eh::OCL_SAFE_CALL(clEnqueueWriteBuffer(queueHolder.get(), memAsHolder.get(), CL_TRUE, 0, sizeBuff, as.data(),
                                           0, nullptr, nullptr));

    eh::OCL_SAFE_CALL(clEnqueueWriteBuffer(queueHolder.get(), memBsHolder.get(), CL_TRUE, 0, sizeBuff, bs.data(),
                                           0, nullptr, nullptr));

    eh::OCL_SAFE_CALL(clEnqueueWriteBuffer(queueHolder.get(), memCsHolder.get(), CL_TRUE, 0, sizeBuff, cs.data(),
                                           0, nullptr, nullptr));

    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
        // std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель

    const char *ptr_kernel_sources = kernel_sources.c_str();
    helpers::Holder<cl_program, int (*)(_cl_program *)>
            programHolder(clCreateProgramWithSource(contextHolder.get(), 1, &ptr_kernel_sources, nullptr, &errcode_ret),
                          clReleaseProgram);

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram

//    eh::OCL_SAFE_CALL(clBuildProgram(programHolder.get(), 1, &selectedDevice, nullptr, nullptr, nullptr));
    errcode_ret = clBuildProgram(programHolder.get(), 1, &selectedDevice, nullptr, nullptr, nullptr);

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    if (errcode_ret == CL_BUILD_PROGRAM_FAILURE) {
        size_t log_size = 0;
        eh::OCL_SAFE_CALL(clGetProgramBuildInfo(programHolder.get(), selectedDevice, CL_PROGRAM_BUILD_LOG, 0, nullptr,
                                                &log_size));
        std::vector<char> log(log_size, 0);
        eh::OCL_SAFE_CALL(clGetProgramBuildInfo(programHolder.get(), selectedDevice, CL_PROGRAM_BUILD_LOG, log_size,
                                                log.data(), nullptr));
        if (log_size > 1) {
            std::cout << "Log:" << std::endl;
            std::cout << log.data() << std::endl;
            std::cout << std::endl;
        }
    }
    eh::OCL_SAFE_CALL(errcode_ret);

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects

    helpers::Holder<cl_kernel, int (*)(_cl_kernel *)>
            kernelHolder(clCreateKernel(programHolder.get(), "aplusb", &errcode_ret), clReleaseKernel);
    eh::OCL_SAFE_CALL(errcode_ret);

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        eh::OCL_SAFE_CALL(clSetKernelArg(kernelHolder.get(), i++, sizeof(cl_mem), &memAsHolder.get()));
        eh::OCL_SAFE_CALL(clSetKernelArg(kernelHolder.get(), i++, sizeof(cl_mem), &memBsHolder.get()));
        eh::OCL_SAFE_CALL(clSetKernelArg(kernelHolder.get(), i++, sizeof(cl_mem), &memCsHolder.get()));
        eh::OCL_SAFE_CALL(clSetKernelArg(kernelHolder.get(), i++, sizeof(unsigned int), &sizeBuff));
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // TODO 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число, кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание, что, чтобы дождаться окончания вычислений (чтобы знать, когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 128;
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;// Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;
            eh::OCL_SAFE_CALL(clEnqueueNDRangeKernel(queueHolder.get(), kernelHolder.get(), 1, nullptr,
                                                     &global_work_size, &workGroupSize, 0, nullptr, &event));
            clWaitForEvents(1, &event);
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter

        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // TODO 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << n / t.lapAvg() / 1E9 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3. * sizeBuff / (1 << 30) / t.lapAvg() << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            eh::OCL_SAFE_CALL(clEnqueueReadBuffer(queueHolder.get(), memCsHolder.get(), CL_TRUE, 0, sizeBuff, cs.data(),
                                                  0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << 1. * sizeBuff / t.lapAvg() / (1 << 30) << " GB/s" << std::endl;
    }

//     TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    return 0;
}
