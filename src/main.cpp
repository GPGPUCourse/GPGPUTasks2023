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

size_t safeGetPlatformPropertySize(cl_platform_id platform, cl_device_info paramName) {
    size_t paramSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, paramName, 0, nullptr, &paramSize));
    return paramSize;
}

size_t safeGetDevicePropertySize(cl_device_id device, cl_device_info paramName) {
    size_t paramSize = 0;
    OCL_SAFE_CALL(clGetDeviceInfo(device, paramName, 0, nullptr, &paramSize));
    return paramSize;
}

void safeGetPlatformProperty(cl_platform_id device, cl_platform_info paramName, size_t paramValueSize, void *paramValue, size_t *paramValueSizeRet) {
    OCL_SAFE_CALL(clGetPlatformInfo(device, paramName, paramValueSize, paramValue, paramValueSizeRet));
}

void safeGetDeviceProperty(cl_device_id device, cl_device_info paramName, size_t paramValueSize, void *paramValue, size_t *paramValueSizeRet) {
    OCL_SAFE_CALL(clGetDeviceInfo(device, paramName, paramValueSize, paramValue, paramValueSizeRet));
}

const char *deviceTypeToName(cl_device_type type) {
    switch (type) {
        case CL_DEVICE_TYPE_CPU:
            return "CPU";
            break;
        case CL_DEVICE_TYPE_GPU:
            return "GPU";
            break;
        default:
            return "Other";
    }
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
        size_t platformNameSize = safeGetPlatformPropertySize(platform, CL_PLATFORM_NAME);
        size_t platformVendorSize = safeGetPlatformPropertySize(platform, CL_PLATFORM_VENDOR);
        // TODO 1.1
        // Попробуйте вместо CL_PLATFORM_NAME передать какое-нибудь случайное число - например 239
        // Т.к. это некорректный идентификатор параметра платформы - то метод вернет код ошибки
        // Макрос OCL_SAFE_CALL заметит это, и кинет ошибку с кодом
        // Откройте таблицу с кодами ошибок:
        // libs/clew/CL/cl.h:103
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        // Найдите там нужный код ошибки и ее название
        // === CL_INVALID_VALUE
        // Затем откройте документацию по clGetPlatformInfo и в секции Errors найдите ошибку, с которой столкнулись
        // === CL_INVALID_VALUE if param_name is not one of the supported values or if size in bytes specified by param_value_size is less than size of return type and param_value is not a NULL value.
        // в документации подробно объясняется, какой ситуации соответствует данная ошибка, и это позволит, проверив код, понять, чем же вызвана данная ошибка (некорректным аргументом param_name)
        // Обратите внимание, что в этом же libs/clew/CL/cl.h файле указаны всевоможные defines, такие как CL_DEVICE_TYPE_GPU и т.п.

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<unsigned char> platformName(platformNameSize, 0);
        safeGetPlatformProperty(platform, CL_PLATFORM_NAME, platformName.size(), platformName.data(), &platformNameSize);
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        std::vector<unsigned char> platformVendor(platformVendorSize, 0);
        safeGetPlatformProperty(platform, CL_PLATFORM_VENDOR, platformVendor.size(), platformVendor.data(), &platformVendorSize);
        std::cout << "    Vendor name: " << platformVendor.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "    Number of devices: " << devicesCount << std::endl;

        // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = devices[deviceIndex];

            size_t deviceNameSize = safeGetDevicePropertySize(device, CL_DEVICE_NAME);
            size_t deviceTypeSize = safeGetDevicePropertySize(device, CL_DEVICE_TYPE);
            size_t deviceMemorySize = safeGetDevicePropertySize(device, CL_DEVICE_GLOBAL_MEM_SIZE); 
            size_t deviceVendorSize = safeGetDevicePropertySize(device, CL_DEVICE_VENDOR); 
            size_t deviceDriverVersionSize = safeGetDevicePropertySize(device, CL_DRIVER_VERSION);

            std::vector<unsigned char> deviceName(deviceNameSize, 0);
            cl_device_type deviceType;
            cl_ulong deviceMemory;
            std::vector<unsigned char> deviceVendor(deviceVendorSize, 0);
            std::vector<unsigned char> deviceDriverVersion(deviceDriverVersionSize, 0);

            safeGetDeviceProperty(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), &deviceNameSize);
            safeGetDeviceProperty(device, CL_DEVICE_TYPE, deviceTypeSize, &deviceType, &deviceTypeSize);
            safeGetDeviceProperty(device, CL_DEVICE_GLOBAL_MEM_SIZE, deviceMemorySize, &deviceMemory, &deviceMemorySize);
            safeGetDeviceProperty(device, CL_DEVICE_VENDOR, deviceVendorSize, deviceVendor.data(), &deviceVendorSize);
            safeGetDeviceProperty(device, CL_DRIVER_VERSION, deviceDriverVersionSize, deviceDriverVersion.data(), &deviceDriverVersionSize);

            std::cout << "        Device name: " << deviceName.data() << std::endl;
            std::cout << "        Device type: " << deviceTypeToName(deviceType) << std::endl;
            std::cout << "        Memory size: " << deviceMemory / (1024 * 1024) << " MB" << std::endl;
            std::cout << "        Device vendor: " << deviceVendor.data() << std::endl;
            std::cout << "        Device driver version: " << deviceDriverVersion.data() << std::endl;
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
        }
    }

    return 0;
}
