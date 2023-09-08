#include <CL/cl.h>
#include <libclew/ocl_init.h>

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


std::string getDeviceTypeName(cl_device_type deviceType) {
    std::string deviceTypeStr;
    switch (deviceType) {
        case CL_DEVICE_TYPE_CPU:
            deviceTypeStr = "CPU";
            break;
        case CL_DEVICE_TYPE_GPU:
            deviceTypeStr = "GPU";
            break;
        default:
            deviceTypeStr = "Other";
            break;
    }
    return deviceTypeStr;
}

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");

    // Откройте
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:
    std::vector<cl_platform_id> platforms;
    cl_uint platformsCount = 0;

    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
    platforms.reserve(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        cl_platform_id platform = platforms[platformIndex];

        // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
        // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
        std::vector<unsigned char> platformName;
        {
            size_t platformNameSize = 0;
            OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
            // TODO 1.1
            // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
            // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки (-30)
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
            platformName.resize(platformNameSize, 0);
            // clGetPlatformInfo(...);
            clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformName.size(), platformName.data(), nullptr);
            std::cout << "    Platform name: " << platformName.data() << std::endl;

        }

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        std::vector<unsigned char> platformVendor;
        {
            size_t platformVendorSize = 0;
            clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize);
            platformVendor.resize(platformVendorSize, 0);
            clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr);
        }
        std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        std::vector<cl_device_id> devices;
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        devices.resize(devicesCount, nullptr);
        OCL_SAFE_CALL(
                clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, (cl_uint) (devices.size()), devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            cl_device_id deviceId = devices[deviceIndex];
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount
                      << " (Id " << deviceId << ")" << std::endl;

            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            std::vector<unsigned char> deviceName;
            {
                size_t deviceNameSize = 0;
                OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
                deviceName.resize(deviceNameSize, 0);
                OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, deviceName.size(), deviceName.data(),
                                              nullptr));
            }
            std::cout << "        Name: " << deviceName.data() << std::endl;

            cl_device_type deviceType;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
            std::string deviceTypeStr = getDeviceTypeName(deviceType);
            std::cout << "        Type: " << deviceTypeStr << std::endl;

            cl_ulong deviceLocalMemSize;
            cl_ulong deviceGlobalMemSize;
            cl_ulong deviceGlobalMemCacheSize;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId,
                                          CL_DEVICE_LOCAL_MEM_SIZE,
                                          sizeof(deviceLocalMemSize),
                                          &deviceLocalMemSize,
                                          nullptr));
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId,
                                          CL_DEVICE_GLOBAL_MEM_SIZE,
                                          sizeof(deviceGlobalMemSize),
                                          &deviceGlobalMemSize,
                                          nullptr));
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId,
                                          CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                                          sizeof(deviceGlobalMemCacheSize),
                                          &deviceGlobalMemCacheSize,
                                          nullptr));
            std::cout << "        Memory size:" << std::endl;
            std::cout << "            Local: " << deviceLocalMemSize / 1024 << " KiB" << std::endl;
            std::cout << "            Global: " << deviceGlobalMemSize / 1024 / 1024.f << " MiB" << std::endl;
            std::cout << "            Cache: " << deviceGlobalMemCacheSize / 1024 << " KiB" << std::endl;

//            cl_device_id deviceParentDevice;
//            OCL_SAFE_CALL(clGetDeviceInfo(deviceId,
//                                          CL_DEVICE_PARENT_DEVICE, // apparently does not exists?
//                                          sizeof(deviceParentDevice),
//                                          &deviceParentDevice,
//                                          nullptr));
//
//            std::cout << "         Parent: " << deviceGlobalMemCacheSize / 1024 / 1024.f << " Mb" << std::endl;

            cl_bool deviceCompilerAvailable;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId,
                                          CL_DEVICE_COMPILER_AVAILABLE, // apparently does not exists?
                                          sizeof(deviceCompilerAvailable),
                                          &deviceCompilerAvailable,
                                          nullptr));

            std::cout << "         Compiler available: " << (deviceCompilerAvailable ? "true" : "false") << std::endl;
        }
    }

    return 0;
}