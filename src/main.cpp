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

static constexpr double gibi = 1024.0 * 1024.0 * 1024.0;

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
#define OCL_CHECK_EC(errcode) OCL_SAFE_CALL(errcode)

/// \return chosen platform, or null
static cl_platform_id chooseAnyPlatform() {
    cl_platform_id platform = nullptr;
    OCL_SAFE_CALL(clGetPlatformIDs(1, &platform, nullptr));
    return platform;
}

/// \return chosen device, or null
static cl_device_id chooseSuitableDevice(cl_platform_id platform) {
    cl_device_id device = nullptr;
    if (!clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, nullptr) && device)
        return device;
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, nullptr));
    return device;
}

template<typename T>
static T getPlainDeviceInfo(cl_device_id device, cl_device_info paramName) {
    T result;
    OCL_SAFE_CALL(clGetDeviceInfo(device, paramName, sizeof result, &result, nullptr));
    return result;
}

struct CLContext {
    cl_context context = nullptr;

    CLContext(cl_platform_id platform, cl_device_id device) {
        struct {
            cl_context_properties key;
            cl_platform_id value;
            cl_context_properties end;
        } platformProperty{CL_CONTEXT_PLATFORM, platform, 0};
        cl_int errorcode;
        context = clCreateContext(
                /* properties = */ &platformProperty.key,
                /* num_devices = */ 1,
                /* devices = */ &device,
                /* pfn_notify = */ nullptr,
                /* user_data = */ nullptr,
                /* errcode_ret = */ &errorcode);
        OCL_CHECK_EC(errorcode);
    }

    ~CLContext() {
        if (context)
            // can fail, but we ignore it
            clReleaseContext(context);
    }
};

struct CLCommandQueue {
    cl_command_queue commandQueue = nullptr;

    CLCommandQueue(CLContext &context, cl_device_id device) {
        cl_int errorcode;
        commandQueue = clCreateCommandQueue(context.context, device,
                                            /* properties = */ 0, &errorcode);
        OCL_CHECK_EC(errorcode);
    }

    ~CLCommandQueue() {
        if (commandQueue)
            // can fail, but we ignore it
            clReleaseCommandQueue(commandQueue);
    }
};

struct CLBuffer {
    cl_mem buffer = nullptr;
    size_t size = 0;

    static CLBuffer createCopiedReadOnly(CLContext &context, void *data, size_t size) {
        std::cerr << size << std::endl;
        CLBuffer result;
        result.size = size;
        cl_int errorcode;
        result.buffer =
                clCreateBuffer(context.context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size, data, &errorcode);
        OCL_CHECK_EC(errorcode);
        return result;
    }

    template<typename T, typename = std::enable_if<std::is_pod<T>::value>>
    static CLBuffer createCopiedReadOnly(CLContext &context, const std::vector<T> &vector) {
        return createCopiedReadOnly(context, (void *) vector.data(), vector.size() * sizeof(T));
    }

    static CLBuffer createWriteOnly(CLContext &context, size_t size) {
        CLBuffer result;
        result.size = size;
        cl_int errorcode;
        result.buffer = clCreateBuffer(context.context, CL_MEM_WRITE_ONLY, size, nullptr, &errorcode);
        OCL_CHECK_EC(errorcode);
        return result;
    }

    ~CLBuffer() {
        if (buffer)
            // can fail, but we ignore it
            clReleaseMemObject(buffer);
    }
};

struct CLProgram {
    cl_program program = nullptr;

    static CLProgram createWithSource(CLContext &context, const std::string &source) {
        CLProgram result;
        cl_int errorcode;
        const char *data = source.data();
        size_t size = source.size();
        result.program = clCreateProgramWithSource(context.context, 1, &data, &size, &errorcode);
        OCL_CHECK_EC(errorcode);
        return result;
    }

    void build(cl_device_id device) {
        OCL_SAFE_CALL(clBuildProgram(program,
                                     /* num_devices = */ 1,
                                     /* devices = */ &device,
                                     /* options = */ "",
                                     /* pfn_notify = */ nullptr,
                                     /* user_data = */ nullptr));
    }

    void dumpBuildLog(cl_device_id device) {
        size_t logSize;
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                            /* param_value_size = */ 0,
                                            /* param_value = */ nullptr,
                                            /* param_value_size_ret = */ &logSize));
        std::string log;
        log.resize(logSize);
        OCL_SAFE_CALL(clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                                            /* param_value_size = */ log.size(),
                                            /* param_value = */ const_cast<char *>(log.data()),
                                            /* param_value_size_ret = */ nullptr));
        std::cout << "Build log:\n" << log << std::endl;
    }

    ~CLProgram() {
        if (program)
            // can fail, but we ignore it
            clReleaseProgram(program);
    }
};

struct CLKernel {
    cl_kernel kernel = nullptr;
    size_t lastArgIndex = 0;

    CLKernel(CLProgram &program, const char *kernelName) {
        cl_int errorcode;
        kernel = clCreateKernel(program.program, kernelName, &errorcode);
        OCL_CHECK_EC(errorcode);
    }

    void appendArg(CLBuffer &buffer) {
        OCL_SAFE_CALL(clSetKernelArg(kernel, lastArgIndex++, sizeof(&buffer.buffer), &buffer.buffer));
    }

    template<typename T, typename = std::enable_if<std::is_pod<T>::value>>
    void appendArg(T t) {
        OCL_SAFE_CALL(clSetKernelArg(kernel, lastArgIndex++, sizeof(T), &t));
    }

    ~CLKernel() {
        if (kernel)
            // can fail, but we ignore ii
            clReleaseKernel(kernel);
    }
};

static cl_event enqueue1DRangeKernel(CLCommandQueue &commandQueue, CLKernel &kernel, size_t globalWorkSize,
                                     size_t localWorkSize) {
    assert(globalWorkSize % localWorkSize == 0);
    cl_event ret;
    OCL_SAFE_CALL(clEnqueueNDRangeKernel(commandQueue.commandQueue, kernel.kernel,
                                         /* work_dim = */ 1,
                                         /* global_work_offset = */ nullptr, &globalWorkSize, &localWorkSize,
                                         /* num_events_in_wait_list = */ 0,
                                         /* event_wait_list = */ nullptr, &ret));
    return ret;
}

static void waitForEvent(cl_event event) {
    OCL_SAFE_CALL(clWaitForEvents(1, &event));
}

/// \pre \p out must have the necessary size allocated
template<typename T, typename = std::enable_if<std::is_pod<T>::value>>
static void readBuffer(CLCommandQueue &commandQueue, CLBuffer &buffer, std::vector<T> &out) {
    OCL_SAFE_CALL(clEnqueueReadBuffer(commandQueue.commandQueue, buffer.buffer,
                                      /* blocking_read = */ CL_TRUE,
                                      /* offset = */ 0,
                                      /* size = */ out.size() * sizeof(T),
                                      /* ptr = */ out.data(),
                                      /* num_events_in_wait_list = */ 0,
                                      /* event_wait_list = */ nullptr,
                                      /* event = */ nullptr));
}

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    cl_platform_id platform = chooseAnyPlatform();
    if (!platform)
        throw std::runtime_error("No platform!");
    cl_device_id device = chooseSuitableDevice(platform);
    if (!device)
        throw std::runtime_error("No suitable device!");

    cl_ulong deviceMaxMemAllocSize = getPlainDeviceInfo<cl_ulong>(device, CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    std::cout << "Max mem alloc size for device: " << deviceMaxMemAllocSize << std::endl;

    // 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    CLContext context(platform, device);

    // 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)
    CLCommandQueue commandQueue(context, device);

    unsigned n = 1000 * 1000 * 100;
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

    // 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)
    CLBuffer bufferA = CLBuffer::createCopiedReadOnly(context, as);
    CLBuffer bufferB = CLBuffer::createCopiedReadOnly(context, bs);
    CLBuffer bufferC = CLBuffer::createWriteOnly(context, n * sizeof(float));
    std::cout << "Buffers A, B, C created!" << std::endl;

    // 6 Выполните 5 (реализуйте кернел в src/cl/aplusb.cl)
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

    // 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
    CLProgram program = CLProgram::createWithSource(context, kernel_sources);

    // 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    program.build(device);
    std::cout << "Program has been built!" << std::endl;

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    program.dumpBuildLog(device);
    std::cout << std::endl;

    // 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    CLKernel kernel(program, "aplusb");

    // 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    kernel.appendArg(bufferA);
    kernel.appendArg(bufferB);
    kernel.appendArg(bufferC);
    kernel.appendArg(n);

    // 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // 12 Запустите выполнения кернела:
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
        for (unsigned i = 0; i < 20; ++i) {
            cl_event event = enqueue1DRangeKernel(commandQueue, kernel, global_work_size, workGroupSize);
            waitForEvent(event);
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        std::cout << "Kernel average time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;

        // 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "GFlops: " << n / t.lapAvg() / 1e9 << std::endl;

        // 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3 * n * sizeof(float) / t.lapAvg() / gibi << " GB/s" << std::endl;
    }

    // 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned i = 0; i < 20; ++i) {
            readBuffer(commandQueue, bufferC, cs);
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << n * sizeof(float) / t.lapAvg() / gibi << " GB/s" << std::endl;
    }

    // 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    return 0;
}
