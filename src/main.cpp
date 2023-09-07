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

std::string stringFrom(cl_device_type type) {
    switch (type) {
    case CL_DEVICE_TYPE_GPU: return "gpu";
    case CL_DEVICE_TYPE_CPU: return "cpu";
    case CL_DEVICE_TYPE_ACCELERATOR: return "accelerator";
    case CL_DEVICE_TYPE_DEFAULT: return "default";
    default: return "unknown";
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

// it is done:
//        OCL_SAFE_CALL(clGetPlatformInfo(platform, 100, 0, nullptr, &platformNameSize));

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        //        std::vector<unsigned char> platformName(platformNameSize, 0);
        const size_t dataSize = 1000;
        char data[dataSize];
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, dataSize, data, NULL));
        std::cout << "    Platform name: " << data << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, dataSize, data, NULL));
        printf("    Platform vendor: %s\n", data);

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, NULL, &devicesCount));

        cl_device_id *devices = (cl_device_id*)calloc(devicesCount, sizeof(cl_device_id));
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices, NULL));

        for (int i = 0; i < devicesCount; ++i) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            printf("device: #%d/%d\n", i + 1, devicesCount);

            OCL_SAFE_CALL(clGetDeviceInfo(devices[i], CL_DEVICE_NAME, dataSize, data, NULL));
            printf("    name: %s\n", data);

            cl_device_type type;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[i], CL_DEVICE_TYPE, sizeof type, &type, NULL));
            printf("    type: %s\n", stringFrom(type).data());

            OCL_SAFE_CALL(clGetDeviceInfo(devices[i], CL_DEVICE_VENDOR, dataSize, data, NULL));
            printf("    vendor: %s\n", data);

            cl_ulong size;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[i], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof size, &size, NULL));
            printf("    size: %lu MB\n", size / 1024 / 1024);

            cl_bool isAvailable;
            OCL_SAFE_CALL(clGetDeviceInfo(devices[i], CL_DEVICE_AVAILABLE, sizeof isAvailable, &isAvailable, NULL));
            printf("    available: %s\n", isAvailable ? "true" : "false");

        }

        free(devices);
    }

    return 0;
}
