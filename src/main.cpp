#include "ClWrappers/Buffer.h"
#include "ClWrappers/CommandQueue.h"
#include "ClWrappers/Context.h"
#include "ClWrappers/Device.h"
#include "ClWrappers/Kernel.h"
#include "ClWrappers/Program.h"

#include "ClWrappers/common.h"

#include <cmath>
#include <iostream>

using namespace ClWrappers;

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    //  1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)

    Device device(0);
    for (auto platform : GetVector<cl_platform_id, cl_uint>(clGetPlatformIDs)) {
        for (auto device_id : GetVector<cl_device_id, cl_uint>(clGetDeviceIDs, platform, CL_DEVICE_TYPE_ALL)) {
            device = Device(device_id);
            if (device.GetDeviceType() & CL_DEVICE_TYPE_GPU) {
                break;
            }
        }
        if (device && device.GetDeviceType() & CL_DEVICE_TYPE_GPU) {
            break;
        }
    }

    if (!device) {
        std::cerr << "Device not found" << std::endl;
        return 1;
    }

    std::cout << "Selected device: " << device.GetDeviceInfo() << std::endl;

    //  2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    Context context(nullptr, 1, &(device.GetDeviceId()), nullptr, nullptr);

    //  3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)
    CommandQueue commandQueue(context, device);

    unsigned int n = 64 * 1000 * 1000;
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

    //  4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)
    auto memorySize = n * sizeof(float);
    auto isCPU = device.GetDeviceType() & CL_DEVICE_TYPE_CPU;
    auto readMode = CL_MEM_READ_ONLY | (isCPU ? CL_MEM_USE_HOST_PTR : CL_MEM_COPY_HOST_PTR);
    auto aBuffer = Buffer(context, readMode, memorySize, as.data());
    auto bBuffer = Buffer(context, readMode, memorySize, bs.data());
    auto cBuffer = Buffer(context,
                          CL_MEM_WRITE_ONLY | (isCPU ? CL_MEM_USE_HOST_PTR : 0),
                          memorySize,
                          isCPU ? cs.data() : nullptr);

    //  6 Выполните  5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
    }

    //  7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
    const char* data[] = {kernel_sources.data()};
    size_t length[] = {kernel_sources.size()};
    Program program(context, sizeof(data) / sizeof(char*), data, length);

    //  8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    const cl_device_id devices[] = {device};
    program.Build(sizeof(devices) / sizeof(cl_device_id), devices, "");

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    auto log = program.GetBuildLog(device);
    if (log.size() > 1) {
        std::cerr << "Log:" << std::endl;
        std::cerr << log.data() << std::endl;
    }

    //  9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Kernel Objects
    auto kernel = Kernel(program, "aplusb");

    //  10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        kernel.SetArg<cl_mem>(i++, aBuffer);
        kernel.SetArg<cl_mem>(i++, bBuffer);
        kernel.SetArg<cl_mem>(i++, cBuffer);
        kernel.SetArg(i++, n);
    }

    //  11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    //  12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число, кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание, что, чтобы дождаться окончания вычислений (чтобы знать, когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 64; // Must be less than or equal to the corresponding values specified by CL_DEVICE_MAX_WORK_ITEM_SIZES[0]
        size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;
        timer t;// Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        for (unsigned int i = 0; i < 20; ++i) {
            cl_event clEvent;
            OCL_SAFE_CALL(
                    clEnqueueNDRangeKernel(commandQueue, kernel, 1,
                                           nullptr, &global_work_size, &workGroupSize,
                                           0, nullptr, &clEvent));
            OCL_SAFE_CALL(
                    clWaitForEvents(1, &clEvent));
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        //  13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " <<  n * 1e-9 / t.lapAvg() << std::endl;

        //  14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3 * memorySize / t.lapAvg() / (1<<30) << " GB/s" << std::endl;
    }

    for (unsigned int i = 0; i < n; ++i) {
        as[i] += bs[i];
    }

    //  15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(
                        clEnqueueReadBuffer(commandQueue, cBuffer, true,
                                            0, memorySize, isCPU ? bs.data() : cs.data(),
                                            0, nullptr, nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << memorySize / t.lapAvg() / (1 << 30) << " GB/s" << std::endl;
    }

    //  16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (std::abs(cs[i] - as[i]) > 1e-4) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    return 0;
}
