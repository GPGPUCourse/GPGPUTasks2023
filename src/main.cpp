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
#include <type_traits>
#include <unordered_map>

//#define DEBUG

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

template <typename RES, typename... ARGS>
requires(std::is_fundamental_v<RES>)
RES OCL_CALL(std::string&& filename, int line, auto&& func, ARGS&&... args)  {
    RES result;
    reportError(std::forward<decltype(func)>(func)(std::forward<ARGS>(args)..., sizeof(result), &result, nullptr), filename, line);
    return result;
}

template <typename RES, typename SIZE, typename... ARGS>
requires(requires(RES&& tmp) { tmp.size(); tmp.data(); })
RES OCL_CALL(std::string&& filename, int line, auto&& func, ARGS&&... args) {
    SIZE size = 0;
    reportError(std::forward<decltype(func)>(func)(std::forward<ARGS>(args)..., 0, nullptr, &size), filename, line);
    RES result(size, 0);
    reportError(std::forward<decltype(func)>(func)(std::forward<ARGS>(args)..., size, result.data(), nullptr), filename, line);
    return result;
}

#define OCL_SAFE_CALL_OLD(expr) reportError(expr, __FILE__, __LINE__)
#define OCL_SAFE_CALL(RES, ...) OCL_CALL<RES>(__FILE__, __LINE__, __VA_ARGS__)
#define OCL_SAFE_CALL_ARRAY(RES, SIZE, ...) OCL_CALL<RES, SIZE>(__FILE__, __LINE__, __VA_ARGS__)

std::vector<cl_device_id> getDevicesByPlatform(cl_platform_id platform, int type) {
    std::vector<cl_device_id> result;
    for (cl_device_id device : OCL_SAFE_CALL_ARRAY(std::vector<cl_device_id>, cl_uint, clGetDeviceIDs, platform, CL_DEVICE_TYPE_ALL)) {
        auto deviceType = OCL_SAFE_CALL(cl_device_type, clGetDeviceInfo, device, CL_DEVICE_TYPE);
        if (deviceType & type)
            result.push_back(device);
    }
    return result;
}

class Context {
private:
    cl_context ctx;
    cl_device_id device;

public:
    explicit Context(cl_device_id dev)
        : device(dev)
    {
        cl_int err;
        ctx = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &err);
        if (CL_SUCCESS != err)
            throw std::runtime_error("Cannot create CL Context with error: " + std::to_string(err));
    }

    [[nodiscard]] cl_device_id getDevice() const {
        return device;
    }

    [[nodiscard]] cl_context getClContext() const {
        return ctx;
    }

    ~Context() {
        cl_int err = clReleaseContext(ctx);
        if (CL_SUCCESS != err)
            std::cerr << "Cannot release CL Context with error: " + std::to_string(err) << std::endl;
    }
};

class CommandQueue {
private:
    cl_command_queue queue;

public:
    explicit CommandQueue(const Context& ctx) {
        cl_int err;
        queue = clCreateCommandQueue(ctx.getClContext(), ctx.getDevice(), 0, &err);
        if (CL_SUCCESS != err)
            throw std::runtime_error("Cannot create CL Command Queue with error: " + std::to_string(err));
    }

    [[nodiscard]] cl_command_queue getClQueue() const {
        return queue;
    }

    ~CommandQueue() {
        cl_int err = clReleaseCommandQueue(queue);
        if (CL_SUCCESS != err)
            std::cerr << "Cannot release CL Command Queue with error: " + std::to_string(err) << std::endl;
    }
};

class Buffer {
private:
    cl_mem mem = nullptr;

public:
    Buffer(const Context& ctx, cl_mem_flags flags, size_t size, void* host_ptr) {
        cl_int err;
        mem = clCreateBuffer(ctx.getClContext(), flags, size, host_ptr, &err);
        if (CL_SUCCESS != err)
            throw std::runtime_error("Cannot create CL Buffer with error: " + std::to_string(err));
    }

    template <typename T>
    Buffer(const Context& ctx, cl_mem_flags flags, const std::vector<T>& data) : Buffer(ctx, flags, sizeof(T) * data.size(), (void*)data.data())
    {}

    [[nodiscard]] cl_mem getClMem() const {
        return mem;
    }

    [[nodiscard]] const cl_mem* getClMemPtr() const {
        return &mem;
    }

    ~Buffer() {
        cl_int err = clReleaseMemObject(mem);
        if (CL_SUCCESS != err)
            std::cerr << "Cannot release CL Buffer with error: " + std::to_string(err) << std::endl;
    }
};

class Program {
private:
    cl_program prog;
    const Context* ctx;

public:
    Program(const Context& context, const std::string& source) : ctx(&context) {
        cl_int err;
        const char* ptrSource = source.c_str();
        prog = clCreateProgramWithSource(ctx->getClContext(), 1, &ptrSource, nullptr, &err);
        if (CL_SUCCESS != err)
            throw std::runtime_error("Cannot create CL program with error: " + std::to_string(err));
    }

    [[nodiscard]] cl_program getClProgram() const { return prog; }

    [[nodiscard]] std::string build(const std::string& options = "") const {
        cl_int err = clBuildProgram(prog, 0, nullptr, options.data(), nullptr, nullptr);
        auto log = OCL_SAFE_CALL_ARRAY(std::string, size_t, clGetProgramBuildInfo, prog, ctx->getDevice(), CL_PROGRAM_BUILD_LOG);

        if (CL_SUCCESS != err) {
#ifdef DEBUG
            if (log.size() > 1) {
                std::cout << "Log:" << std::endl;
                std::cout << log.data() << std::endl;
            }
#endif
            throw std::runtime_error("Cannot build CL program with error: " + std::to_string(err));
        }

//        cl_build_status status = OCL_SAFE_CALL(cl_build_status, clGetProgramBuildInfo, prog, ctx->getDevice(), CL_PROGRAM_BUILD_STATUS);
        return log;
    }

    ~Program() {
        cl_int err = clReleaseProgram(prog);
        if (CL_SUCCESS != err)
            std::cerr << "Cannot release CL program with error: " + std::to_string(err) << std::endl;
    }
};

class Kernel {
private:
    cl_kernel kernel;

public:
    Kernel(const Program& prog, const std::string& name) {
        cl_int err;
        kernel = clCreateKernel(prog.getClProgram(), name.data(), &err);
        if (CL_SUCCESS != err)
            throw std::runtime_error("Cannot create CL kernel with error: " + std::to_string(err));
    }

    [[nodiscard]] cl_kernel getClKernel() const { return kernel; }

    ~Kernel() {
        cl_int err = clReleaseKernel(kernel);
        if (CL_SUCCESS != err)
            std::cerr <<"Cannot release CL kernel with error: " + std::to_string(err) << std::endl;
    }
};

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // TODO 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    auto platforms = OCL_SAFE_CALL_ARRAY(std::vector<cl_platform_id>, cl_uint, clGetPlatformIDs);

    cl_platform_id platform = nullptr;
    const std::string pltName = "NVIDIA";

    for (cl_platform_id plt : platforms) {
        auto platformName = OCL_SAFE_CALL_ARRAY(std::string, size_t, clGetPlatformInfo, plt, CL_PLATFORM_NAME);
        if (platformName.find(pltName) != std::string::npos)
            platform = plt;
    }
    if (!platform) platform = platforms.front();

    std::vector<cl_device_id> devices = getDevicesByPlatform(platform, 0xFFFF);
    cl_device_id deviceId = devices.front();
    auto deviceName = OCL_SAFE_CALL_ARRAY(std::string, size_t, clGetDeviceInfo, deviceId, CL_DEVICE_NAME);
    std::cout << "Device: " << deviceName.data() << std::endl;

    auto deviceType = OCL_SAFE_CALL(cl_device_type, clGetDeviceInfo, deviceId, CL_DEVICE_TYPE);
    std::cout << "Device type: ";
    if (deviceType & CL_DEVICE_TYPE_CPU)         std::cout << "CPU ";
    if (deviceType & CL_DEVICE_TYPE_GPU)         std::cout << "GPU ";
    if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) std::cout << "ACCELERATOR ";
    if (deviceType & CL_DEVICE_TYPE_DEFAULT)     std::cout << "DEFAULT TYPE ";
    if (!(deviceType & (CL_DEVICE_TYPE_CPU | CL_DEVICE_TYPE_GPU | CL_DEVICE_TYPE_ACCELERATOR | CL_DEVICE_TYPE_DEFAULT))) {
        std::cout << "SOMETHING STRANGE";
    }
    std::cout << std::endl;

    // TODO 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    Context ctx(deviceId);

    // TODO 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)
    CommandQueue cmdQueue(ctx);

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


    cl_mem_flags flags = (deviceType == CL_DEVICE_TYPE_CPU) ? CL_MEM_USE_HOST_PTR : CL_MEM_COPY_HOST_PTR;
    Buffer a(ctx, CL_MEM_READ_ONLY | flags, as);
    Buffer b(ctx, CL_MEM_READ_ONLY | flags, bs);
    Buffer c(ctx, CL_MEM_WRITE_ONLY | flags, cs);

    // TODO 6 Выполните TODO 5 (реализуйте кернел в src/cl/aplusb.cl)
    // затем убедитесь, что выходит загрузить его с диска (убедитесь что Working directory выставлена правильно - см. описание задания),
    // напечатав исходники в консоль (if проверяет, что удалось считать хоть что-то)
    std::string kernel_sources;
    {
        std::ifstream file("src/cl/aplusb.cl");
        kernel_sources = std::string(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
        if (kernel_sources.empty()) {
            throw std::runtime_error("Empty source file! May be you forgot to configure working directory properly?");
        }
#ifdef DEBUG
        std::cout << kernel_sources << std::endl;
#endif
    }

    // TODO 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
    Program program(ctx, kernel_sources);

    // TODO 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    std::string log = program.build();
    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo

    // TODO 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    Kernel kernel(program, "aplusb");

    // TODO 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
         unsigned int i = 0;
         OCL_SAFE_CALL_OLD(clSetKernelArg(kernel.getClKernel(), i++, sizeof(cl_mem), a.getClMemPtr()));
         OCL_SAFE_CALL_OLD(clSetKernelArg(kernel.getClKernel(), i++, sizeof(cl_mem), b.getClMemPtr()));
         OCL_SAFE_CALL_OLD(clSetKernelArg(kernel.getClKernel(), i++, sizeof(cl_mem), c.getClMemPtr()));
         OCL_SAFE_CALL_OLD(clSetKernelArg(kernel.getClKernel(), i++, sizeof(unsigned int), &n));
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
            OCL_SAFE_CALL_OLD(clEnqueueNDRangeKernel(cmdQueue.getClQueue(),
                                   kernel.getClKernel(),
                                   1,
                                   nullptr,
                                   &global_work_size,
                                   &workGroupSize,
                                   0,
                                   nullptr,
                                   &event));
            OCL_SAFE_CALL_OLD(clWaitForEvents(1, &event));
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
        std::cout << "GFlops: " << (float)n / (float)1'000'000'000 / t.lapAvg() << std::endl;

        // TODO 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        std::cout << "VRAM bandwidth: " << 3.0*n*sizeof(float) / (1 << 30) / t.lapAvg() << " GB/s" << std::endl;
    }

    // TODO 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL_OLD(clEnqueueReadBuffer(cmdQueue.getClQueue(),
                                c.getClMem(),
                                CL_TRUE,
                                0,
                                n * sizeof(float),
                                cs.data(),
                                0,
                                nullptr,
                                nullptr));
            t.nextLap();
        }
        std::cout << "Result data transfer time: " << t.lapAvg() << "+-" << t.lapStd() << " s" << std::endl;
        std::cout << "VRAM -> RAM bandwidth: " << (float)n*sizeof(float) / (float)(1 << 30) / t.lapAvg() << " GB/s" << std::endl;
    }

    // TODO 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    return 0;
}
