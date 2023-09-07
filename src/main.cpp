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

#define OCL_SAFE_CALL(expr) reportError((expr), __FILE__, __LINE__)


int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init()) {
        throw std::runtime_error("Can't init OpenCL driver!");
    }

    // Откройте
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    // Прочитайте документацию clGetPlatformIDs и убедитесь, что этот способ узнать, сколько есть платформ, соответствует документации:

    // Цитата из документации:
    // Parameters
    // num_entries
    //     The number of cl_platform_id entries that can be added to platforms.
    //     If platforms is not NULL, the num_entries must be greater than zero.
    // platforms
    //     Returns a list of OpenCL platforms found.
    //     The cl_platform_id values returned in platforms can be used to identify a specific OpenCL platform.
    //     If platforms argument is NULL, this argument is ignored.
    //     The number of OpenCL platforms returned is the mininum of the value specified
    //     by num_entries or the number of OpenCL platforms available.
    // num_platforms
    //     Returns the number of OpenCL platforms available.
    //     If num_platforms is NULL, this argument is ignored.
    //
    // Таким образом, вызов метода с аргументами 0, nullptr и &platformsCount
    // поместит количество доступных платформ в platformsCount.
    cl_uint platformsCount = 0;
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Тот же метод используется для того, чтобы получить идентификаторы всех платформ - сверьтесь с документацией, что это сделано верно:

    // По той же цитате из документации, передаём количество платформ
    // и указатель на массив их идентификаторов.
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

        // OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));
        // Получил ошибку CL_INVALID_VALUE (-30), что означает либо некорректный param_name,
        // либо param_value_size меньше размера возвращаемого значения, указатель на место для которого
        // param_value при этом не nullptr. Он nullptr, поэтому только первый вариант.

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
#define GET_INFO_ARRAY(func, type, vecName, id, param)                                                                       \
    size_t vecName##Size = 0;                                                                                          \
    OCL_SAFE_CALL(func(id, param, 0, nullptr, &vecName##Size));                                                        \
    std::vector<type> vecName(vecName##Size);                                                                          \
    OCL_SAFE_CALL(func(id, param, vecName##Size, vecName.data(), nullptr));

        GET_INFO_ARRAY(clGetPlatformInfo, char, vendorName, platform, CL_PLATFORM_VENDOR)
        std::cout << "    Platform vendor: " << vendorName.data() << '\n';

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> deviceIdx(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, deviceIdx.data(), nullptr));

        if (devicesCount == 0) {
            std::cout << "    No devices on the platform\n";
        }

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            std::cout << "    Device #" << deviceIndex + 1 << "/" << devicesCount << '\n';
            cl_device_id deviceId = deviceIdx[deviceIndex];

            GET_INFO_ARRAY(clGetDeviceInfo, char, deviceName, deviceId, CL_DEVICE_NAME);
            std::cout << "        Device name: " << deviceName.data() << '\n';

#define SINGLE_INFO(var) sizeof(var), &var, nullptr

            cl_device_type deviceType = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, SINGLE_INFO(deviceType)));
            std::cout << "        Device type: " << deviceType << '\n';

            cl_ulong deviceMemBytes = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, SINGLE_INFO(deviceMemBytes)));
            cl_ulong deviceMemMB = deviceMemBytes / 1024 / 1024;
            std::cout << "        Device memory: " << deviceMemMB << " MB\n";

            cl_bool deviceAvailable = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_AVAILABLE, SINGLE_INFO(deviceAvailable)));
            std::cout << "        Device available? " << (deviceAvailable ? "YES" : "NO") << '\n';

            cl_bool deviceEndianLittle = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_ENDIAN_LITTLE, SINGLE_INFO(deviceEndianLittle)));
            std::cout << "        Device little endian? " << (deviceEndianLittle ? "YES" : "NO") << '\n';
        }
    }

    return 0;
}
