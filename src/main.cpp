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
        //
        // OCL_SAFE_CALL(clGetPlatformInfo(platform, 239, 0, nullptr, &platformNameSize));
        // 
        // Result of changing CL_PLATFORM_NAME to 239:
        //      terminate called after throwing an instance of 'std::runtime_error'
        //       what():  OpenCL error code -30 encountered at .../GPGPUTasks2023/src/main.cpp:56
        // Corresponds to CL_INVALID_VALUE
        

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        std::vector<unsigned char> platformName(platformNameSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
        std::cout << "    Platform name: " << platformName.data() << std::endl;

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        // MY std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;
        size_t platformVendorSize = 0;
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, 0, nullptr, &platformVendorSize));

        std::vector<unsigned char> platformVendor(platformVendorSize, 0);
        OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_VENDOR, platformVendorSize, platformVendor.data(), nullptr));
        std::cout << "    Platform vendor: " << platformVendor.data() << std::endl;

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            std::cout <<"    Device #" << (deviceIndex+1) << "/" << devicesCount << std::endl;
            cl_device_id device = devices[deviceIndex];

            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            size_t deviceNameSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
            std::vector<unsigned char> deviceName(deviceNameSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
            std::cout << "        Device name: " << deviceName.data() << std::endl;

            // - Тип устройства (видеокарта/процессор/что-то странное)
            cl_device_type device_type;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type, nullptr));
            std::cout << "        Device type: ";
            switch (device_type) {
                case CL_DEVICE_TYPE_CPU:
                    std::cout << "CPU";
                    break;
                case CL_DEVICE_TYPE_GPU:
                    std::cout << "GPU";
                    break;
                case CL_DEVICE_TYPE_ACCELERATOR:
                    std::cout << "Accelerator";
                    break;
                case CL_DEVICE_TYPE_DEFAULT:
                    std::cout << "Default";
                    break;
                // NOTE: CL_DEVICE_TYPE_CUSTOM is not declared in cl.h, however it is
                // declared as an option by the documentation. I ignore it.
                default:
                    std::cout << "Couldn't understand. Possibly combination of several or CUSTOM.";
                    break;
            }
            std::cout << std::endl;

            // - Размер памяти устройства в мегабайтах
            cl_ulong memorySize;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), &memorySize, nullptr));
            std::cout << "        Device memory: " << (memorySize / (1<<20)) <<" megabytes" << std::endl;

            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными
            // 1. Version of OpenCL. I have OpenCL 2.1 (Build 0)
            size_t deviceVersionSize = 0;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, 0, nullptr, &deviceVersionSize));
            std::vector<unsigned char> deviceVersion(deviceVersionSize, 0);
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_VERSION, deviceVersionSize, deviceVersion.data(), nullptr));
            std::cout << "        OpenCL version: " << deviceVersion.data() << std::endl;

            // 2. Maximum work group size. I have 8192
            size_t maxWorkGroupSize;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &maxWorkGroupSize, nullptr));
            std::cout << "        Maximum work group size: " << maxWorkGroupSize << std::endl;

            // 3. images are supported by the OpenCL device. True for me
            cl_bool imageSupport;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &imageSupport, nullptr));
            std::cout << "        Images are supported? " << imageSupport << std::endl;

            // 4. Number of compute units. I have 12
            // NOTE: for some reason glitches from time to time and prints gibberish like 140582869532684.
            cl_uint maxComputeUnits;
            OCL_SAFE_CALL(clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_uint), &maxComputeUnits, nullptr));
            std::cout << "        Compute units on the OpenCL device: " << maxComputeUnits << std::endl;
        }
    }

    return 0;
}
