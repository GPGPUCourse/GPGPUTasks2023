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
    std::vector <cl_platform_id> platforms(platformsCount);
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
        std::vector<unsigned char> platformName(platformNameSize + 1, 0);
        // clGetPlatformInfo(...);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        size_t vendorNameSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &vendorNameSize));
        std::vector<unsigned char> vendorName(vendorNameSize + 1, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, vendorNameSize, vendorName.data(), nullptr));
        std::cout << "    Vendor name: " << vendorName.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector <cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        size_t infoSize = 0;
        std::vector<unsigned char> deviceInfo;
        deviceInfo.assign(256, 0);

        std::cout << "    Number of Devices: " << devicesCount << std::endl;
        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            auto device_id = devices[deviceIndex];
            std::cout << "        Device(" << device_id << "): #" << deviceIndex << '/' << devicesCount << std::endl;

            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_NAME, 0, nullptr, &infoSize));
            deviceInfo.resize(infoSize + 1, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_NAME, infoSize, deviceInfo.data(), nullptr));
            std::cout << "            NAME: " << deviceInfo.data() << std::endl;

            cl_device_type device_type;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_TYPE, sizeof(device_type), &device_type, nullptr));
            std::cout << "            TYPE: ";
            if (device_type & CL_DEVICE_TYPE_CPU)
                std::cout << "CL_DEVICE_TYPE_CPU ";
            if (device_type & CL_DEVICE_TYPE_GPU)
                std::cout << "CL_DEVICE_TYPE_GPU ";
            if (device_type & CL_DEVICE_TYPE_ACCELERATOR)
                std::cout << "CL_DEVICE_TYPE_ACCELERATOR ";
            if (device_type & CL_DEVICE_TYPE_DEFAULT)
                std::cout << "CL_DEVICE_TYPE_DEFAULT ";
            std::cout << std::endl;

            uint64_t global_memory_size = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(global_memory_size),
                                          &global_memory_size, nullptr));
            std::cout << "            GLOBAL_MEM_SIZE: " << (global_memory_size >> 20) << " MB" << std::endl;

            uint64_t global_memory_cache = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE, sizeof(global_memory_cache),
                                          &global_memory_cache, nullptr));
            std::cout << "            GLOBAL_MEM_CACHE_SIZE: " << global_memory_cache << " B" << std::endl;

            cl_uint max_clock_frequency = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_CLOCK_FREQUENCY, sizeof(max_clock_frequency),
                                          &max_clock_frequency, nullptr));
            std::cout << "            MAX_CLOCK_FREQUENCY: " << max_clock_frequency << " MHz" << std::endl;

            cl_uint max_compute_units = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(max_compute_units),
                                          &max_compute_units, nullptr));
            std::cout << "            MAX_COMPUTE_UNITS: " << max_compute_units << std::endl;
        }
    }

    return 0;
}
