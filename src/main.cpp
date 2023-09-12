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

// Немного упростим себе жизнь при вызовах OpenCL API
// W от слова Wrapper, т.е. обёртка

// unique_ptr для ресурсов OpenCL
template<typename T, CL_API_ENTRY cl_int(*Destructor) CL_API_CALL(T), T NullVal = nullptr>
class WBox
{
    T mElem;

public:
    WBox() noexcept : WBox(NullVal) {
    }
    WBox(T elem) noexcept : mElem(elem) {
    }
    ~WBox() {
        Clear();
    }

    WBox(const WBox &) = delete;
    WBox &operator=(const WBox &) = delete;

    WBox(WBox &&rhs) noexcept : mElem(rhs.Release()) {
    }

    WBox &operator=(WBox &&rhs) noexcept {
        if (this == &rhs) {
            return *this;
        }
        Clear();
        mElem = rhs.Release();
        return *this;
    }

    T Get() const noexcept {
        return mElem;
    }

    T Release() noexcept {
        T retVal = mElem;
        mElem = NullVal;
        return retVal;
    }

    void Clear() {
        if (mElem) {
            if (Destructor) {
                OCL_SAFE_CALL(Destructor(mElem));
            }
            mElem = NullVal;
        }
    }
};

template<typename Size, typename ElemPtr, typename Elem, typename... Args>
class WArrayGetter
{
    using Getter = cl_int (*)(Args..., Size, ElemPtr, Size *);
    Getter mGetter;

public:
    explicit WArrayGetter(Getter getter) noexcept : mGetter(getter) {
    }

    std::vector<Elem> operator()(Args... args) const {
        Size count = 0;
        OCL_SAFE_CALL(mGetter(args..., 0, nullptr, &count));
        std::vector<Elem> result(count);
        OCL_SAFE_CALL(mGetter(args..., count, &result[0], nullptr));
        // Полагаемся на RVO
        return result;
    }
};

template<typename Elem, typename... Args>
using WVecGetter = WArrayGetter<cl_uint, Elem *, Elem, Args...>;

template<typename Char, typename... Args>
using WStringGetter = WArrayGetter<size_t, void *, Char, Args...>;

const WVecGetter<cl_platform_id> GetPlatformIDs(&clGetPlatformIDs);
const WVecGetter<cl_device_id, cl_platform_id, cl_device_type> GetDeviceIDs(&clGetDeviceIDs);
const WStringGetter<char, cl_platform_id, cl_platform_info> GetPlatformString(&clGetPlatformInfo);
const WStringGetter<char, cl_device_id, cl_device_info> GetDeviceString(&clGetDeviceInfo);
const WStringGetter<char, cl_program, cl_device_id, cl_program_build_info> GetProgramBuildInfo(clGetProgramBuildInfo);

void SelectDevice(cl_platform_id *pOutPlatformId, cl_device_id *pOutDeviceId) {
    cl_device_type bestType = CL_DEVICE_TYPE_ALL;
    cl_ulong bestMem = 0;

    std::vector<cl_platform_id> platformIdx = GetPlatformIDs();
    for (cl_platform_id platformId : platformIdx) {
        std::vector<cl_device_id> deviceIdx = GetDeviceIDs(platformId, CL_DEVICE_TYPE_ALL);
        for (cl_device_id deviceId : deviceIdx) {
            cl_device_type deviceType = 0;
            cl_ulong deviceMem = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(deviceMem), &deviceMem, nullptr));

            bool betterOption = bestMem == 0;
            const cl_device_type IDEAL_TYPE = CL_DEVICE_TYPE_GPU;
            betterOption |= deviceType == IDEAL_TYPE && bestType != IDEAL_TYPE;
            betterOption |= deviceMem > bestMem;
            betterOption &= !(deviceType != IDEAL_TYPE && bestType == IDEAL_TYPE);
            if (!betterOption) {
                continue;
            }

            *pOutPlatformId = platformId;
            *pOutDeviceId = deviceId;
            bestType = deviceType;
            bestMem = deviceMem;
        }
    }
}

class WContext
{
    WBox<cl_context, clReleaseContext> mContext;

    static void CL_CALLBACK HandleNotify(const char *errInfo, const void *privateInfo, size_t cb, void *userData) {
        std::cerr << "[CL][err] " << errInfo << std::endl;
    }

public:
    explicit WContext(cl_device_id deviceId) {
        cl_int errCode = 0;
        mContext = clCreateContext(nullptr, 1, &deviceId, &HandleNotify, nullptr, &errCode);
        OCL_SAFE_CALL(errCode);
    }

    cl_context Get() const noexcept {
        return mContext.Get();
    }
};

class WCommandQueue
{
    WBox<cl_command_queue, clReleaseCommandQueue> mQueue;

public:
    explicit WCommandQueue(cl_context context, cl_device_id deviceId, cl_command_queue_properties properties) {
        cl_int errCode = 0;
        mQueue = clCreateCommandQueue(context, deviceId, properties, &errCode);
        OCL_SAFE_CALL(errCode);
    }

    cl_command_queue Get() const noexcept {
        return mQueue.Get();
    }
};

class WBuffer
{
    WBox<cl_mem, clReleaseMemObject> mBuffer;

public:
    explicit WBuffer(cl_context context, cl_mem_flags flags, size_t size, void *pHost) {
        cl_int errCode = 0;
        mBuffer = clCreateBuffer(context, flags, size, pHost, &errCode);
        OCL_SAFE_CALL(errCode);
    }

    cl_mem Get() const noexcept {
        return mBuffer.Get();
    }
};

class WProgram
{
    WBox<cl_program, clReleaseProgram> mProgram;
    cl_device_id mDeviceId = nullptr;

public:
    explicit WProgram(cl_context context, const std::string &source) {
        const char *strings[] = {source.c_str()};
        const size_t lengths[] = {source.size()};
        cl_int errCode = 0;
        mProgram = clCreateProgramWithSource(context, 1, strings, lengths, &errCode);
        OCL_SAFE_CALL(errCode);
    }

    cl_program Get() const noexcept {
        return mProgram.Get();
    }

    void Build(cl_device_id deviceId, const char *options) {
        mDeviceId = deviceId;
        OCL_SAFE_CALL(clBuildProgram(Get(), 1, &deviceId, options, nullptr, nullptr));
    }

    void PrintBuildLog() {
        std::vector<char> buildLog = GetProgramBuildInfo(Get(), mDeviceId, CL_PROGRAM_BUILD_LOG);
        if (buildLog.size() <= 1) {
            std::cout << "Build log is empty\n";
        } else {
            std::cout << "Build log:\n" << &buildLog[0] << std::endl;
        }
    }
};

class WKernel
{
    WBox<cl_kernel, clReleaseKernel> mKernel;

public:
    explicit WKernel(cl_program program, const char *kernelName) {
        cl_int errCode = 0;
        mKernel = clCreateKernel(program, kernelName, &errCode);
        OCL_SAFE_CALL(errCode);
    }

    cl_kernel Get() const noexcept {
        return mKernel.Get();
    }

    template<typename T>
    void SetArg(cl_uint argIndex, const T &arg) {
        OCL_SAFE_CALL(clSetKernelArg(Get(), argIndex, sizeof(T), &arg));
    }
};

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // DONE 1 По аналогии с предыдущим заданием узнайте, какие есть устройства, и выберите из них какое-нибудь
    // (если в списке устройств есть хоть одна видеокарта - выберите ее, если нету - выбирайте процессор)
    cl_platform_id platformId = nullptr;
    cl_device_id deviceId = nullptr;
    SelectDevice(&platformId, &deviceId);

    std::vector<char> platformName = GetPlatformString(platformId, CL_PLATFORM_NAME);
    std::vector<char> deviceName = GetDeviceString(deviceId, CL_DEVICE_NAME);
    std::cout << "Selected platform " << &platformName[0] << " and device " << &deviceName[0] << std::endl;

    // DONE 2 Создайте контекст с выбранным устройством
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Contexts -> clCreateContext
    // Не забывайте проверять все возвращаемые коды на успешность (обратите внимание, что в данном случае метод возвращает
    // код по переданному аргументом errcode_ret указателю)
    // И хорошо бы сразу добавить в конце clReleaseContext (да, не очень RAII, но это лишь пример)
    WContext context(deviceId);

    // DONE 3 Создайте очередь выполняемых команд в рамках выбранного контекста и устройства
    // См. документацию https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/ -> OpenCL Runtime -> Runtime APIs -> Command Queues -> clCreateCommandQueue
    // Убедитесь, что в соответствии с документацией вы создали in-order очередь задач
    // И хорошо бы сразу добавить в конце clReleaseQueue (не забывайте освобождать ресурсы)
    WCommandQueue commandQueue(context.Get(), deviceId, 0);

    size_t n = 100 * 1000 * 1000;
    // Создаем два массива псевдослучайных данных для сложения и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);
    FastRandom r(n);
    for (size_t i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }
    std::cout << "Data generated for n=" << n << "!" << std::endl;

    // DONE 4 Создайте три буфера в памяти устройства (в случае видеокарты - в видеопамяти - VRAM) - для двух суммируемых массивов as и bs (они read-only) и для массива с результатом cs (он write-only)
    // См. Buffer Objects -> clCreateBuffer
    // Размер в байтах соответственно можно вычислить через sizeof(float)=4 и тот факт, что чисел в каждом массиве n штук
    // Данные в as и bs можно прогрузить этим же методом, скопировав данные из host_ptr=as.data() (и не забыв про битовый флаг, на это указывающий)
    // или же через метод Buffer Objects -> clEnqueueWriteBuffer
    // И хорошо бы сразу добавить в конце clReleaseMemObject (аналогично, все дальнейшие ресурсы вроде OpenCL под-программы, кернела и т.п. тоже нужно освобождать)
    WBuffer bufA(context.Get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), &as[0]);
    WBuffer bufB(context.Get(), CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, n * sizeof(float), &bs[0]);
    WBuffer bufC(context.Get(), CL_MEM_WRITE_ONLY, n * sizeof(float), nullptr);

    // DONE 6 Выполните DONE 5 (реализуйте кернел в src/cl/aplusb.cl)
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

    // DONE 7 Создайте OpenCL-подпрограмму с исходниками кернела
    // см. Runtime APIs -> Program Objects -> clCreateProgramWithSource
    // у string есть метод c_str(), но обратите внимание, что передать вам нужно указатель на указатель
    WProgram program(context.Get(), kernel_sources);

    // DONE 8 Теперь скомпилируйте программу и напечатайте в консоль лог компиляции
    // см. clBuildProgram
    program.Build(deviceId, "");

    // А также напечатайте лог компиляции (он будет очень полезен, если в кернеле есть синтаксические ошибки - т.е. когда clBuildProgram вернет CL_BUILD_PROGRAM_FAILURE)
    // Обратите внимание, что при компиляции на процессоре через Intel OpenCL драйвер - в логе указывается, какой ширины векторизацию получилось выполнить для кернела
    // см. clGetProgramBuildInfo
    //    size_t log_size = 0;
    //    std::vector<char> log(log_size, 0);
    //    if (log_size > 1) {
    //        std::cout << "Log:" << std::endl;
    //        std::cout << log.data() << std::endl;
    //    }
    program.PrintBuildLog();

    // DONE 9 Создайте OpenCL-kernel в созданной подпрограмме (в одной подпрограмме может быть несколько кернелов, но в данном случае кернел один)
    // см. подходящую функцию в Runtime APIs -> Program Objects -> Kernel Objects
    WKernel kernel(program.Get(), "aplusb");

    // DONE 10 Выставите все аргументы в кернеле через clSetKernelArg (as_gpu, bs_gpu, cs_gpu и число значений, убедитесь, что тип количества элементов такой же в кернеле)
    {
        cl_uint i = 0;
        kernel.SetArg(i++, bufA.Get());
        kernel.SetArg(i++, bufB.Get());
        kernel.SetArg(i++, bufC.Get());
        kernel.SetArg(i++, cl_ulong(n));
    }

    // DONE 11 Выше увеличьте n с 1000*1000 до 100*1000*1000 (чтобы дальнейшие замеры были ближе к реальности)

    // DONE 12 Запустите выполнения кернела:
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
        for (int i = 0; i < 20; ++i) {
            // Здесь не так удобно делать обёртку
            cl_event execEvent = nullptr;
            OCL_SAFE_CALL(clEnqueueNDRangeKernel(commandQueue.Get(), kernel.Get(), 1, nullptr, &global_work_size,
                                                 &workGroupSize, 0, nullptr, &execEvent));
            OCL_SAFE_CALL(clWaitForEvents(1, &execEvent));
            OCL_SAFE_CALL(clReleaseEvent(execEvent));
            t.nextLap();// При вызове nextLap секундомер запоминает текущий замер (текущий круг) и начинает замерять время следующего круга
        }
        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам, а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered
        // P.S. чтобы в CLion быстро перейти к символу (функции/классу/много чему еще), достаточно нажать Ctrl+Shift+Alt+N -> lapsFiltered -> Enter
        double avgTime = t.lapAvg();
        std::cout << "Kernel average time: " << avgTime << "+-" << t.lapStd() << " s" << std::endl;

        // DONE 13 Рассчитайте достигнутые гигафлопcы:
        // - Всего элементов в массивах по n штук
        // - Всего выполняется операций: операция a+b выполняется n раз
        // - Флопс - это число операций с плавающей точкой в секунду
        // - В гигафлопсе 10^9 флопсов
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        double avgGFlops = 1e-9 * n / avgTime;// NOLINT
        std::cout << "GFlops: " << avgGFlops << std::endl;

        // DONE 14 Рассчитайте используемую пропускную способность обращений к видеопамяти (в гигабайтах в секунду)
        // - Всего элементов в массивах по n штук
        // - Размер каждого элемента sizeof(float)=4 байта
        // - Обращений к видеопамяти 2*n*sizeof(float) байт на чтение и 1*n*sizeof(float) байт на запись, т.е. итого 3*n*sizeof(float) байт
        // - В гигабайте 1024*1024*1024 байт
        // - Среднее время выполнения кернела равно t.lapAvg() секунд
        size_t transferredBytes = 3 * n * sizeof(float);
        double bandwidthGBps = transferredBytes / (1024 * 1024 * 1024 * avgTime);// NOLINT
        std::cout << "VRAM bandwidth: " << bandwidthGBps << " GB/s" << std::endl;
    }

    // DONE 15 Скачайте результаты вычислений из видеопамяти (VRAM) в оперативную память (RAM) - из cs_gpu в cs (и рассчитайте скорость трансфера данных в гигабайтах в секунду)
    {
        size_t bufSize = n * sizeof(float);
        timer t;
        for (int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(commandQueue.Get(), bufC.Get(), CL_TRUE, 0, bufSize, &cs[0], 0, nullptr,
                                              nullptr));
            t.nextLap();
        }
        double avgTime = t.lapAvg();
        std::cout << "Result data transfer time: " << avgTime << "+-" << t.lapStd() << " s" << std::endl;
        double bandwidthGBps = bufSize / (1024 * 1024 * 1024 * avgTime);// NOLINT
        std::cout << "VRAM -> RAM bandwidth: " << bandwidthGBps << " GB/s" << std::endl;
    }

    // DONE 16 Сверьте результаты вычислений со сложением чисел на процессоре (и убедитесь, что если в кернеле сделать намеренную ошибку, то эта проверка поймает ошибку)
    for (size_t i = 0; i < n; ++i) {
        float expected = as[i] + bs[i];
        float actual = cs[i];
        float absDiff = abs(expected - actual);
        bool matches = absDiff < 1e-5;
        if (!matches) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    return 0;
}
