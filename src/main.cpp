#include <CL/cl.h>
#include <libclew/ocl_init.h>

#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "error_print.h"

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
    std::string message = "OpenCL error code " + to_string(err) + " (" + get_opencl_err_string(err) +
                          ") encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)

template<class ParamT, class ObjT, class OcvFuncT, class ParamProviderT, class ParamToRawConverterT>
ParamT getInfo(ObjT object, cl_device_info param, OcvFuncT ocvFunc, ParamProviderT paramProvider,
               ParamToRawConverterT convertToRawType) {
    size_t size = 0;
    OCL_SAFE_CALL(ocvFunc(object, param, 0, nullptr, &size));
    ParamT paramValue = paramProvider(size);
    OCL_SAFE_CALL(ocvFunc(object, param, size, convertToRawType(paramValue), nullptr));
    return paramValue;
}

template<class ParamT, class ObjT, class OcvFuncT>
ParamT getInfo(ObjT object, cl_device_info param, OcvFuncT ocvFunc) {
    ParamT paramValue;
    OCL_SAFE_CALL(ocvFunc(object, param, sizeof(ParamT), &paramValue, nullptr));
    return paramValue;
}

template<class ObjT, class OcvFuncT>
std::string getStringInfo(ObjT object, cl_device_info param, OcvFuncT ocvFunc) {
    return getInfo<std::string>(
            object, param, ocvFunc, [](size_t size) { return std::string(size, 0); },
            [](std::string &str) { return (void *) &str[0]; });
}

std::string getVendorName(cl_platform_id platform) {
    return getStringInfo(platform, CL_PLATFORM_VENDOR, clGetPlatformInfo);
}

std::vector<cl_device_id> getDevices(cl_platform_id platform) {
    cl_uint devicesCount = 0;
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
    std::vector<cl_device_id> devices(devicesCount);
    OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
    return devices;
}

std::string getDeviceStringInfo(cl_device_id device, cl_device_info param) {
    return getStringInfo(device, param, clGetDeviceInfo);
}

template<class T>
T getDeviceInfo(cl_device_id device, cl_device_info param) {
    return getInfo<T>(device, param, clGetDeviceInfo);
}

std::string getDeviceType(cl_device_id device) {
    auto type = getDeviceInfo<cl_device_type>(device, CL_DEVICE_TYPE);
    std::string res;
    if (type & CL_DEVICE_TYPE_CPU)
        res += "CPU ";
    if (type & CL_DEVICE_TYPE_GPU)
        res += "GPU ";
    if (type & CL_DEVICE_TYPE_ACCELERATOR)
        res += "ACCELERATOR ";
    if (res.empty())
        return "UNKNOWN";
    return res;
}

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        size_t platformNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
        // TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
        // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        std::cout << "    Vendor name: " << getVendorName(platform) << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        std::vector<cl_device_id> devices = getDevices(platform);
        std::cout << "    Devices:" << std::endl;
        for (int deviceIndex = 0; deviceIndex < devices.size(); ++deviceIndex) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            cl_device_id device = devices[deviceIndex];
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devices.size() << std::endl;
            std::cout << "        Device name: " << getDeviceStringInfo(device, CL_DEVICE_NAME) << std::endl;
            std::cout << "        Device type: " << getDeviceType(device) << std::endl;
            auto globalMemSize = getDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_SIZE);
            std::cout << "        Global memory size: " << globalMemSize << " (" << (globalMemSize >> 20) << " MB)"
                      << std::endl;
            std::cout << "        Global memory cache size: "
                      << getDeviceInfo<cl_ulong>(device, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE) << std::endl;
            std::cout << "        Global memory cache line size: "
                      << getDeviceInfo<cl_uint>(device, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE) << std::endl;
            std::cout << "        Clock frequency: " << getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_CLOCK_FREQUENCY)
                      << std::endl;
            std::cout << "        Compute units: " << getDeviceInfo<cl_uint>(device, CL_DEVICE_MAX_COMPUTE_UNITS)
                      << std::endl;
            std::cout << "        Work group max size: " << getDeviceInfo<size_t>(device, CL_DEVICE_MAX_WORK_GROUP_SIZE)
                      << std::endl;
        }
    }

    return 0;
}
