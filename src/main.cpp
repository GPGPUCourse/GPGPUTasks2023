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

std::string checkDeviceType(cl_device_type deviceType)
{
    if (deviceType & CL_DEVICE_TYPE_GPU) {
        return "GPU";
    } else if (deviceType & CL_DEVICE_TYPE_CPU) {
        return "CPU";
    } else if (deviceType & CL_DEVICE_TYPE_ACCELERATOR) {
        return "ACCELERATOR";
    } else {
        return "UNKNOWN";
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
        size_t vendorNameSize = 0;
        cl_int result = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize);
        if (result != CL_SUCCESS) {
            throw std::runtime_error("Failed to get platform vendor name size, error code: " + std::to_string(result));
        }
        std::vector<unsigned char> platformVendor(vendorNameSize);
        result = clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendor.size(), platformVendor.data(), nullptr);
        if (result != CL_SUCCESS) {
            throw std::runtime_error("Failed to get platform vendor name, error code: " + std::to_string(result));
        }
        std::cout << "        Vendor name: " << platformVendor.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::cout << "        Number of devices: " << devicesCount << std::endl;

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout << "        Device #" << deviceIndex + 1 << "/" << devicesCount << std::endl;
            cl_device_id device = devices[deviceIndex];
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
            std::vector<unsigned char> deviceName(deviceNameSize);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
            std::cout << "            Device Name: " << deviceName.data() << std::endl;

            // - Тип устройства (видеокарта/процессор/что-то странное)
            size_t deviceTypeSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, 0, nullptr, &deviceTypeSize));
            if (deviceTypeSize > sizeof(cl_device_type)) {
                throw std::runtime_error("Size mismatch: required " + to_string(deviceTypeSize) + "; actual: " + to_string(sizeof(cl_device_type)));
            }

            cl_device_type deviceType = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, deviceTypeSize, &deviceType, nullptr));
            std::cout << "            Device Type: " << checkDeviceType(deviceType) << std::endl;

            // - Размер памяти устройства в мегабайтах
            size_t deviceMemSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, 0, nullptr, &deviceMemSize));
            if (deviceMemSize > sizeof(cl_ulong)) {
                throw std::runtime_error("Size mismatch: required " + to_string(deviceMemSize) + "; actual: " + to_string(sizeof(cl_ulong)));
            }

            cl_ulong deviceMem = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, deviceMemSize, &deviceMem, nullptr));
            std::cout << "            Device Global Memory (MB): " << deviceMem / 1e6 << std::endl;

            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            // -- Количество compute units устройства
            size_t deviceUnitSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, 0, nullptr, &deviceUnitSize));
            if (deviceUnitSize > sizeof(cl_uint))
            {
                throw std::runtime_error("Size mismatch: required " + to_string(deviceUnitSize) + "; actual: " + to_string(sizeof(cl_uint)));
            }

            cl_uint deviceUnits = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, deviceUnitSize, &deviceUnits, nullptr));
            std::cout << "            Number of compute units: " << deviceUnits << std::endl;

            // -- Размер локальной памяти устройства
            size_t deviceLocalMemSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_LOCAL_MEM_SIZE, 0, nullptr, &deviceLocalMemSize));
            if (deviceLocalMemSize > sizeof(cl_ulong))
            {
                throw std::runtime_error("Size mismatch: required " + to_string(deviceLocalMemSize) + "; actual: " + to_string(sizeof(cl_ulong)));
            }

            cl_ulong deviceLocalMemory = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, deviceLocalMemSize, &deviceLocalMemory, nullptr));
            std::cout << "            Device Local Memory (MB): " << deviceLocalMemory / 1e6 << std::endl;
        }
    }

    return 0;
}
