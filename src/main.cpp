#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>


template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


int main() {
    cl_int error_code = 0;

    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)

    cl_uint nPlatforms = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &nPlatforms));
    std::cout << nPlatforms << " platforms available." << std::endl;

    std::vector<cl_platform_id> platforms(nPlatforms, 0);
    OCL_SAFE_CALL(clGetPlatformIDs(nPlatforms, platforms.data(), nullptr));

    cl_device_id bestDeviceId = 0;
    cl_platform_id bestPlatformId = 0;
    bool gpuDeviceFound = false, deviceFound = false;

    for (int i = 0; i < nPlatforms; i++) {
        size_t platformNameSize = 0;
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 0, nullptr, &platformNameSize);
        std::vector<unsigned char> platformName(platformNameSize, 0);
        clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr);

        std::cout << "Platform #" << i + 1 << "/" << nPlatforms << ":" << std::endl;
        std::cout << "    Platform's name: " << platformName.data() << std::endl;

        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "    Platform's devices found: " << devicesCount << std::endl;

        std::vector<cl_device_id> platformDevices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL, devicesCount, platformDevices.data(), nullptr));

        std::map<cl_device_type, std::string> deviceTypeMap = {
            { (cl_device_type)(1 << 0), std::string("DEFAULT") },
            { (cl_device_type)(1 << 1), std::string("CPU") },
            { (cl_device_type)(1 << 2), std::string("GPU") },
            { (cl_device_type)(1 << 3), std::string("ACCELERATOR") },
        };

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "    Device #" << deviceIndex + 1 << "/" << devicesCount << std::endl;
            cl_device_id device = platformDevices[deviceIndex];

            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
            std::vector<unsigned char> deviceName(platformNameSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
            std::cout << "        Device name: " << deviceName.data() << std::endl;

            // Тип
            size_t deviceTypeSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, 0, nullptr, &deviceTypeSize));
            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, deviceTypeSize, &deviceType, nullptr));
            std::string deviceTypeStr = deviceTypeMap.find(deviceType) == deviceTypeMap.end() ? "Unknown" : deviceTypeMap[deviceType];
            std::cout << "        Device type: " << deviceTypeStr << std::endl;

            if (!gpuDeviceFound && (deviceTypeMap.find(deviceType) != deviceTypeMap.end()) && deviceTypeMap[deviceType] == "GPU") {
                bestDeviceId = device;
                bestPlatformId = platforms[i];
                deviceFound = true;
                gpuDeviceFound = true;
            }
            else if (!deviceFound && (deviceTypeMap.find(deviceType) != deviceTypeMap.end()) && deviceTypeMap[deviceType] == "CPU") {
                bestDeviceId = device;
                bestPlatformId = platforms[i];
                deviceFound = true;
            }
        }
    }
    if (!deviceFound) {
        std::cout << "No GPU or CPU device found" << std::endl;
        return 1;
    }


    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    cl_context_properties props[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties)(bestPlatformId), 0 };
    cl_device_id devices[1] = { bestDeviceId };
    cl_context context;
    context = clCreateContext(props, 1, devices, nullptr, nullptr, &error_code);
    OCL_SAFE_CALL(error_code);
    if (!context)
    {
        printf("Error: Failed to create a compute context!\n");
        return 1;
    }
    std::cout << "Context created successfully" << std::endl;

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)

    cl_command_queue queue;
    queue = clCreateCommandQueue(context, bestDeviceId, 0, &error_code);
    OCL_SAFE_CALL(error_code);
    std::cout << "Queue created successfully" << std::endl;

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

    // TODO 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)

    cl_mem_flags readBufferFlags = 0, writeBufferFlags = 0;
    writeBufferFlags |= CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR;
    if (gpuDeviceFound) {
        readBufferFlags |= CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR;
    } else {
        readBufferFlags |= CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR;
    }

    cl_mem bufferA, bufferB, bufferC;
    bufferA = clCreateBuffer(context, readBufferFlags, sizeof(float) * n, as.data(), &error_code);
    OCL_SAFE_CALL(error_code);
    bufferB = clCreateBuffer(context, readBufferFlags, sizeof(float) * n, bs.data(), &error_code);
    OCL_SAFE_CALL(error_code);

    bufferC = clCreateBuffer(context, writeBufferFlags, sizeof(float) * n, nullptr, &error_code);
    OCL_SAFE_CALL(error_code);

    std::cout << "Buffers created successfully" << std::endl;

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
    std::vector<size_t> lengths = { kernel_sources.length() };
    const char* kernel_sources_str = kernel_sources.c_str();

    cl_program program = clCreateProgramWithSource(context, 1, &kernel_sources_str, lengths.data(), &error_code);
    OCL_SAFE_CALL(error_code);

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    error_code = clBuildProgram(program, 1, devices, nullptr, nullptr, nullptr);

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, bestDeviceId, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(program, bestDeviceId, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));
    if (log_size > 1) {
        std::cout << "Log:" << std::endl;
        std::cout << "--------------" << std::endl;
        std::cout << log.data() << std::endl;
        std::cout << "--------------" << std::endl;
    }
    OCL_SAFE_CALL(error_code);

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    cl_kernel kernel = clCreateKernel(program, "aplusb", &error_code);
    OCL_SAFE_CALL(error_code);

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        unsigned int i = 0;
        clSetKernelArg(kernel, i++, sizeof(cl_mem), (void*)(&bufferA));
        clSetKernelArg(kernel, i++, sizeof(cl_mem), (void*)(&bufferB));
        clSetKernelArg(kernel, i++, sizeof(cl_mem), (void*)(&bufferC));
        clSetKernelArg(kernel, i++, sizeof(unsigned int), (void *)(&n));
    }

    // TODO 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)
    // DONE

    // TODO 12 Запустите выполнения кернела:
    // - С одномерной рабочей группой размера 128
    // - В одномерном рабочем пространстве размера roundedUpN, где roundedUpN - наименьшее число, кратное 128 и при этом не меньшее n
    // - см. clEnqueueNDRangeKernel
    // - Обратите внимание, что, чтобы дождаться окончания вычислений (чтобы знать, когда можно смотреть результаты в cs_gpu) нужно:
    //   - Сохранить событие "кернел запущен" (см. аргумент "cl_event *event")
    //   - Дождаться завершения полунного события - см. в документации подходящий метод среди Event Objects
    {
        size_t workGroupSize = 128;
        size_t global_work_size = n % workGroupSize == 0 ? n : n + workGroupSize - n % workGroupSize;

        timer t;// Это вспомогательный секундомер, он замеряет время своего создания и позволяет усреднять время нескольких замеров
        cl_event kernelExecution;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(queue, kernel, 1, nullptr, &global_work_size, nullptr, 0, nullptr, &kernelExecution));
            OCL_SAFE_CALL(clWaitForEvents(1, &kernelExecution));
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
        std::cout << "GFlops: " << (float)n / t.lapAvg() / 1e9 << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3 * (float)n * sizeof(float) / (1024 * 1024 * 1024) / t.lapAvg() << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        cl_event readBuffer;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(queue, bufferC, false, 0, n * sizeof(float), (void*)cs.data(), 0, nullptr, &readBuffer));
            OCL_SAFE_CALL(clWaitForEvents(1, &readBuffer));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << (float)n * sizeof(float) / (1024 * 1024 * 1024) / t.lapAvg() << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
       for (unsigned int i = 0; i < n; ++i) {
           if (cs[i] != as[i] + bs[i]) {
               throw std::runtime_error("CPU and GPU results differ!");
           }
       }

    OCL_SAFE_CALL(clReleaseKernel(kernel));
    OCL_SAFE_CALL(clReleaseProgram(program));
    OCL_SAFE_CALL(clReleaseMemObject(bufferA));
    OCL_SAFE_CALL(clReleaseMemObject(bufferB));
    OCL_SAFE_CALL(clReleaseMemObject(bufferC));
    
    OCL_SAFE_CALL(clReleaseContext(context));
    OCL_SAFE_CALL(clReleaseCommandQueue(queue));

    return 0;
}
