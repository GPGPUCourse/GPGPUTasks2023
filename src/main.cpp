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

void printPlatformInfo(cl_platform_id platform);
std::vector<cl_platform_id> getAndPrintPlatforms(cl_uint& platformsCount);
void printDeviceInfo(cl_device_id deviceId);

int main() {
    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку libs/clew)
    if (!ocl_init())
        throw std::runtime_error("Can't init OpenCL driver!");
    cl_uint platformsCount;
    std::vector<cl_platform_id> platforms = getAndPrintPlatforms(platformsCount);

    for (int platformIndex = 0; platformIndex < platformsCount; ++platformIndex) {
        cl_platform_id platform = platforms[platformIndex];
        std::cout << "Platform #" << (platformIndex + 1) << "/" << platformsCount << std::endl;
        printPlatformInfo(platform);

        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        cl_uint devicesCount = 0;
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, 0, nullptr, &devicesCount));
        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));

        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            cl_device_id deviceId = devices[deviceIndex];
            std::cout << "    Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;

            printDeviceInfo(deviceId);
        }
    }

    return 0;
}

std::vector<cl_platform_id> getAndPrintPlatforms(cl_uint& platformsCount) {
    platformsCount= 0;
    // https://www.khronos.org/registry/OpenCL/sdk/1.2/docs/man/xhtml/
    // Нажмите слева: "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformIDs"
    OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsCount));
    std::cout << "Number of OpenCL platforms: " << platformsCount << std::endl;

    // Тот же метод используется для того, чтобы получить идентификаторы всех платформ
    std::vector<cl_platform_id> platforms(platformsCount);
    OCL_SAFE_CALL(clGetPlatformIDs(platformsCount, platforms.data(), nullptr));
    return std::move(platforms);
}

void printPlatformInfo(cl_platform_id platform) {
    // Откройте документацию по "OpenCL Runtime" -> "Query Platform Info" -> "clGetPlatformInfo"
    // Не забывайте проверять коды ошибок с помощью макроса OCL_SAFE_CALL
    size_t platformNameSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &platformNameSize));
    std::vector<unsigned char> platformName(platformNameSize, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize, platformName.data(), nullptr));
    std::cout << "    Platform name: " << platformName.data() << std::endl;

    size_t vendorNameSize = 0;
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr, &vendorNameSize));
    std::vector<unsigned char> vendorName(vendorNameSize, 0);
    OCL_SAFE_CALL(clGetPlatformInfo(platform, CL_PLATFORM_NAME, vendorNameSize, vendorName.data(), nullptr));
    std::cout << "    Vendor: " << vendorName.data() << std::endl;
}

void printDeviceInfo(cl_device_id deviceId) {
    {
        size_t deviceNameSize = 0;
        OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, 0, nullptr, &deviceNameSize));
        std::vector<unsigned char> deviceName(deviceNameSize, 0);
        OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_NAME, deviceNameSize, deviceName.data(), nullptr));
        std::cout << "        Device name: " << deviceName.data() << std::endl;
    }

    {
        size_t deviceVersionSize = 0;
        OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, 0, nullptr, &deviceVersionSize));
        std::vector<unsigned char> deviceVersion(deviceVersionSize, 0);
        OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_VERSION, deviceVersionSize, deviceVersion.data(), nullptr));
        std::cout << "        Device version: " << deviceVersion.data() << std::endl;
    }

    {
        size_t driverVersionSize = 0;
        OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DRIVER_VERSION, 0, nullptr, &driverVersionSize));
        std::vector<unsigned char> driverVersion(driverVersionSize, 0);
        OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DRIVER_VERSION, driverVersionSize, driverVersion.data(), nullptr));
        std::cout << "        Driver version: " << driverVersion.data() << std::endl;
    }

    {
        cl_device_type deviceType;
        OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_TYPE, sizeof(deviceType), &deviceType, nullptr));
        std::cout << "        Device type: " << deviceType << std::endl;
    }

    {
        cl_ulong memorySize;
        OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(memorySize), &memorySize, nullptr));
        std::cout << "        Memory size: " << memorySize / 1024 / 1024 << " MB" << std::endl;
    }

    {
        size_t maxWorkGroupSize;
        OCL_SAFE_CALL(clGetDeviceInfo(deviceId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(maxWorkGroupSize), &maxWorkGroupSize, nullptr));
        std::cout << "        Max Work Group Size: " << maxWorkGroupSize << std::endl;
    }
}
