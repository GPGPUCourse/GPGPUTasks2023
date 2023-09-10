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


template <typename INFO_SIZE_TYPE, typename CL_GET_INFO_F, typename CL_ID_TYPE,  typename CL_INFO_TYPE>
INFO_SIZE_TYPE getContentSize(CL_GET_INFO_F getInfoFunc, CL_ID_TYPE objId, CL_INFO_TYPE clInfo) {
    INFO_SIZE_TYPE infoSize = 0;
    OCL_SAFE_CALL(getInfoFunc(objId, clInfo, 0, nullptr, &infoSize));
    return infoSize;
}

template <typename RetType, typename CL_GET_INFO_F,typename CL_ID_TYPE, typename CL_INFO_TYPE>
RetType getInfo(CL_GET_INFO_F getInfoFunc, CL_ID_TYPE objId, CL_INFO_TYPE clInfo) {
    auto platformInfoSize = getContentSize<size_t>(getInfoFunc, objId, clInfo);

    RetType platformInfoContent = {};
    OCL_SAFE_CALL(getInfoFunc(objId, clInfo, platformInfoSize, &platformInfoContent, nullptr));
    return platformInfoContent;
}
template <typename CL_GET_INFO_F,typename CL_ID_TYPE, typename CL_INFO_TYPE>
std::string getInfoText(CL_GET_INFO_F getInfoFunc, CL_ID_TYPE objId, CL_INFO_TYPE clInfo) {
    auto platformInfoSize = getContentSize<size_t>(getInfoFunc, objId, clInfo);

    std::vector<unsigned char> platformInfoContent(platformInfoSize - 1, 0);
    OCL_SAFE_CALL(getInfoFunc(objId, clInfo, platformInfoSize, platformInfoContent.data(), nullptr));
    return {platformInfoContent.begin(), platformInfoContent.end()};
}

template <typename RetType, typename CL_GET_INFO_F, typename CL_ID_TYPE,  typename CL_INFO_TYPE>
void prettyPrintInfo(const std::string& outHeader, CL_GET_INFO_F getInfoFunc, CL_ID_TYPE objId, CL_INFO_TYPE clInfo) {
    auto infoContent = getInfo<RetType>(getInfoFunc, objId, clInfo);
    std::string strInfoContent;

    if (infoContent & CL_DEVICE_TYPE_CPU)
        strInfoContent = "cpu";
    else if (infoContent & CL_DEVICE_TYPE_GPU)
        strInfoContent = "gpu";

    if (clInfo == CL_DEVICE_LOCAL_MEM_SIZE)
        strInfoContent = std::to_string((double) infoContent / (1024 * 1024)) + "MB";

    if (strInfoContent.empty())
        strInfoContent = std::to_string(infoContent);

    std::cout <<  outHeader << strInfoContent << std::endl;
}

template <typename CL_GET_INFO_F, typename CL_ID_TYPE,  typename CL_INFO_TYPE>
void prettyPrintInfo(const std::string& outHeader, CL_GET_INFO_F getInfoFunc, CL_ID_TYPE objId, CL_INFO_TYPE clInfo) {
    auto infoContent = getInfoText(getInfoFunc, objId, clInfo);
    std::cout <<  outHeader << infoContent << std::endl;
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

        //ошибка -30 - CL_INVALID_VALUE: param_name is not one of the supported values

        // TODO 1.2
        // Аналогично тому, как был запрошен список идентификаторов всех платформ - так и с названием платформы, теперь, когда известна длина названия - его можно запросить:
        prettyPrintInfo("    Platform name: ", clGetPlatformInfo, platform, CL_PLATFORM_NAME);

        // TODO 1.3
        // Запросите и напечатайте так же в консоль вендора данной платформы
        prettyPrintInfo("    Platform vendor: ", clGetPlatformInfo, platform, CL_PLATFORM_VENDOR);
        // TODO 2.1
        // Запросите число доступных устройств данной платформы (аналогично тому, как это было сделано для запроса числа доступных платформ - см. секцию "OpenCL Runtime" -> "Query Devices")
        auto devicesCount = getContentSize<cl_uint>(clGetDeviceIDs, platform, CL_DEVICE_TYPE_ALL);
        std::cout << "Number of devices available on a platform: " << devicesCount << std::endl;

        std::vector<cl_device_id> devices(devicesCount);
        OCL_SAFE_CALL(clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, devicesCount, devices.data(), nullptr));
        for (int deviceIndex = 0; deviceIndex < devicesCount; ++deviceIndex) {
            // TODO 2.2
            // Запросите и напечатайте в консоль:
            // - Название устройства
            // - Тип устройства (видеокарта/процессор/что-то странное)
            // - Размер памяти устройства в мегабайтах
            // - Еще пару или более свойств устройства, которые вам покажутся наиболее интересными

            std::cout << "Device #" << (deviceIndex + 1) << "/" << devicesCount << std::endl;
            cl_device_id device = devices[deviceIndex];

            prettyPrintInfo("    Device name: ", clGetDeviceInfo, device, CL_DEVICE_NAME);
            prettyPrintInfo<cl_device_type>("    Device type: ", clGetDeviceInfo, device, CL_DEVICE_TYPE);
            prettyPrintInfo<cl_ulong>("    Device memory: ", clGetDeviceInfo, device, CL_DEVICE_LOCAL_MEM_SIZE);
            prettyPrintInfo<cl_ulong>("    Device available: ", clGetDeviceInfo, device, CL_DEVICE_AVAILABLE);
            prettyPrintInfo<cl_ulong>("    Device image support: ", clGetDeviceInfo, device, CL_DEVICE_IMAGE_SUPPORT);
        }
    }

    return 0;
}
