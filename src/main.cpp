#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
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
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
	    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

//     for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
//         std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
//         cl_platform_id platform = platforms[platformIndex];

           
//         size_t platformNameSize = 0;
//         OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
//         std::vector<unsigned char> platformName(platformNameSize, 0);
//         OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), &platformNameSize));
//         std::cout << "    Platform name: " << platformName.data() << std::endl;
        
//         cl_uint devicesCount = 0;
//         OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
//         std::cout << "    Number of platform devices: " << devicesCount << std::endl;
        
//         // Хотим получить идентификаторы всех девайсов
//         std::vector<cl_device_id> devices(devicesCount);
//         OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
        
//         for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
//             std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
//             // TODO 2.2
//             // Запросите и напечатайте в консоль:
//             // - Название устройства
//             // - Тип устройства (видеокарта/процессор/что-то странное)
//             // - Размер памяти устройства в мегабайтах
//             // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            
//             cl_device_id device = devices[deviceIndex];
            
//             // name CL_DEVICE_NAME
//             size_t deviceNameSize = 0;
//             OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
            
//             std::vector<unsigned char> deviceName(deviceNameSize, 0);
//             OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), &deviceNameSize));
//             std::cout << "        Device name: " << deviceName.data() << std::endl;
//         }
//     }
    
    cl_platform_id platform = platforms[1];
    size_t platformNameSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
    std::vector<unsigned char> platformName(platformNameSize, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), &platformNameSize));
    std::cout << "Chosen platform name: " << platformName.data() << std::endl;
    
    cl_uint devicesCount = 0;
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
//     std::cout << "    Number of platform devices: " << devicesCount << std::endl;

    // Хотим получить идентификаторы всех девайсов
    std::vector<cl_device_id> devices(devicesCount);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
        
    cl_device_id device = devices[0];
            
    // name CL_DEVICE_NAME
    size_t deviceNameSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));

    std::vector<unsigned char> deviceName(deviceNameSize, 0);
    OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), &deviceNameSize));
    std::cout << "Chosen device name: " << deviceName.data() << std::endl;
    
    
    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    
    cl_int errcode_ret = 0;
    cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    
    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)

    
    // Флаг CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE я не установил, значит очередь in-order
    errcode_ret = 0;
    cl_command_queue command_queue = clCreateCommandQueue(context, device, 0, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    
    
//     unsigned int n = 1000 * 1000;
    unsigned int n = 100*1000*1000;
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

    size_t buffer_size = sizeof(float) * n;
    cl_mem buffer_as = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size, as.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    
    cl_mem buffer_bs = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, buffer_size, bs.data(), &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);

    cl_mem buffer_cs = clCreateBuffer(context, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, buffer_size, nullptr, &errcode_ret); //?
    OCL_SAFE_CALL(errcode_ret);

        
    
    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        
//         std::ifstream file("/home/boris/Documents/courses_2023_2024/GPGPUTasks2023/src/cl/aplusb.cl"); // !!!
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.size() == 0) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
//         std::cout << kernel_sources << std::endl;
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
        
    
//     std::ifstream file("/home/boris/Documents/courses_2023_2024/GPGPUTasks2023/src/cl/aplusb.cl"); // !!!
    std::ifstream file("src/cl/aplusb.cl");
    cl_uint count = 1;
    std::string s;
    while (getline(file, s)) {
        count += 1;
    }
    
    file.clear();
    file.seekg(0);
    
    std::string file_lines[count];
    for (cl_uint i = 0; i < count; i++) {
        getline(file, file_lines[i]);
    }
    
    const char* strings[count];
    size_t lengths[count];
    for (cl_uint i = 0; i < count; i++) {
        lengths[i] = file_lines[i].size();
        strings[i] = file_lines[i].c_str();
//         std::cout << strings[i] << std::endl;
    }
    
//     for (size_t i = 0; i < count; i++) {
//         std::cout << "line " << strings[i] << std::endl;
//         std::cout << "length " << lengths[i] << std::endl;
//     }
    
//     cl_program subprogram = clCreateProgramWithSource(context, count, lines, lengths, &errcode_ret);
//     OCL_SAFE_CALL(errcode_ret);
    
    cl_program subprogram = clCreateProgramWithSource(context, count, strings, lengths, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    
    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    
    errcode_ret = clBuildProgram(subprogram, 1, &device, nullptr, 0, 0);
    OCL_SAFE_CALL(errcode_ret);

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo

    size_t log_size = 0;
    OCL_SAFE_CALL(clGetProgramBuildInfo(subprogram, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size));
    std::vector<char> log(log_size, 0);
    OCL_SAFE_CALL(clGetProgramBuildInfo(subprogram, device, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr));

    std::cout << "log_size: " << log_size << std::endl;
    
    if (log_size > 0) {
        std::cout << "Log:" << std::endl;
        std::cout << log.data() << std::endl;
    }

        
    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects

    
    const char* kernel_name = "aplusb";
    cl_kernel kernel = clCreateKernel(subprogram, kernel_name, &errcode_ret);
    OCL_SAFE_CALL(errcode_ret);
    
    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    
    unsigned int i = 0;
    OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(buffer_as), &buffer_as));
    OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(buffer_bs), &buffer_bs));
    OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(buffer_cs), &buffer_cs));
    OCL_SAFE_CALL(clSetKernelArg(kernel, i++, sizeof(n), &n));
    

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
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, &global_work_size, &workGroupSize, 0, nullptr, &event));
            OCL_SAFE_CALL(clWaitForEvents(1, &event));
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
        
        double gflops = n * n * t.lapAvg() / 1e+9;
        
        std::cout << "GFlops: " << gflops << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        
        double bandwidth = n * sizeof(float) * 3 * n * sizeof(float) / 1024 / 1024 / 1024 * t.lapAvg();
        
        std::cout << "VRAM bandwidth: " << bandwidth << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(command_queue, buffer_cs, CL_TRUE, 0, buffer_size, cs.data(), 0, nullptr, nullptr));
            OCL_SAFE_CALL(t.nextLap());
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        double bandwidth = n * sizeof(float) * 3 * n * sizeof(float) / 1024 / 1024 / 1024 * t.lapAvg();
        std::cout << "VRAM -> RAM bandwidth: " << bandwidth << " GB/s" << std::endl;
    }

    // Почему Segmentation fault (core dumped)?
    
    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }
    
    // Я проверил, вычисляя cs[global_id] = as[global_id] + bs[global_id] + 1; -- работает

    
    OCL_SAFE_CALL(clReleaseContext(context));
    OCL_SAFE_CALL(clReleaseCommandQueue(command_queue));
    
    OCL_SAFE_CALL(clReleaseMemObject(buffer_as));
    OCL_SAFE_CALL(clReleaseMemObject(buffer_bs));
    OCL_SAFE_CALL(clReleaseMemObject(buffer_cs));
    
    
    
    return 0;
}
