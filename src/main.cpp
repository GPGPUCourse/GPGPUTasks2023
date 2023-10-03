#include <CL/cl.h>
#include <libclew/ocl_init.h>
#include <libutils/fast_random.h>
#include <libutils/timer.h>

#include <cassert>
#include <fstream>
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
    if (CL_SUCCESS == err) {
        return;
    }

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    throw std::runtime_error("OpenCL error code " + to_string(err)
        + " encountered at " + filename + ":" + to_string(line));
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


static cl_device_id select_device() {
    std::vector<cl_platform_id> platforms;

    {
        cl_uint platformsNumber;
        OCL_SAFE_CALL(clGetPlatformIDs(0, nullptr, &platformsNumber));

        platforms.resize(platformsNumber);
        OCL_SAFE_CALL(clGetPlatformIDs(platformsNumber, platforms.data(), nullptr));
    }

    if (platforms.empty()) {
        throw std::runtime_error("no platforms available");
    }

    for (const auto & platform : platforms) {
        std::vector<cl_device_id> devices;

        {
            cl_uint devicesNumber;
            OCL_SAFE_CALL(clGetDeviceIDs(
                platform,
                CL_DEVICE_TYPE_ALL,
                0,
                nullptr,
                &devicesNumber
            ));

            devices.resize(devicesNumber);
            OCL_SAFE_CALL(clGetDeviceIDs(
                platform,
                CL_DEVICE_TYPE_ALL,
                devicesNumber,
                devices.data(),
                nullptr
            ));
        }

        if (devices.empty()) {
            throw std::runtime_error("no devices available");
        }

        cl_device_id fallback = devices[0];
        for (const auto & device : devices) {
            cl_device_type device_type;

            OCL_SAFE_CALL(clGetDeviceInfo(
                device,
                CL_DEVICE_TYPE,
                sizeof(device_type),
                &device_type,
                nullptr
            ));

            if (device_type & CL_DEVICE_TYPE_GPU) {
                return device;
            }

            if (device_type & CL_DEVICE_TYPE_CPU) {
                fallback = device;
            }
        }

        return fallback;
    }

    throw std::runtime_error("unreachable");
}

template<typename T, typename F>
struct holder {

    T value;
    F destructor;

    holder(T value, F destructor)
        : value(std::move(value))
        , destructor(std::move(destructor))
        {}

    ~holder() noexcept {
        destructor(value);
    }

    const T & operator*() const noexcept {
        return value;
    }
};

template<typename T, typename F>
static holder<T, F> make_holder(T && value, F && destructor) {
    return { std::forward<T>(value), std::forward<F>(destructor) };
}

int main() {
    cl_int result = 0;

    // Пытаемся слинковаться с символами OpenCL API в runtime (через библиотеку clew)
    if (!ocl_init()) {
        throw std::runtime_error("Can't init OpenCL driver!");
    }

    const cl_device_id device = select_device();

    const auto ctx = make_holder(
        clCreateContext(
            nullptr,
            1,
            &device,
            nullptr,
            nullptr,
            &result
        ),
        clReleaseContext
    );

    OCL_SAFE_CALL(result);

    const auto cmd_queue = make_holder(
        clCreateCommandQueue(
            *ctx,
            device,
            CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
            &result
        ),
        clReleaseCommandQueue
    );

    OCL_SAFE_CALL(result);

    // const unsigned int n = 1000 * 1000;
    const unsigned int n = 100 * 1000 * 1000;

    // Создаем два массива псевдослучайных данных для сложения
    // и массив для будущего хранения результата
    std::vector<float> as(n, 0);
    std::vector<float> bs(n, 0);
    std::vector<float> cs(n, 0);

    FastRandom r(n);
    for (unsigned int i = 0; i < n; ++i) {
        as[i] = r.nextf();
        bs[i] = r.nextf();
    }

    std::cout << "Data generated for n=" << n << "!" << std::endl;

    const auto as_buf = make_holder(
        clCreateBuffer(
            *ctx,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            sizeof(cl_float) * n,
            as.data(),
            &result
        ),
        clReleaseMemObject
    );

    OCL_SAFE_CALL(result);

    const auto bs_buf = make_holder(
        clCreateBuffer(
            *ctx,
            CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR,
            sizeof(cl_float) * n,
            bs.data(),
            &result
        ),
        clReleaseMemObject
    );

    OCL_SAFE_CALL(result);

    const auto cs_buf = make_holder(
        clCreateBuffer(
            *ctx,
            CL_MEM_WRITE_ONLY | CL_MEM_USE_HOST_PTR,
            sizeof(cl_float) * n,
            cs.data(),
            &result
        ),
        clReleaseMemObject
    );

    OCL_SAFE_CALL(result);

    std::string kernel_sources;

    {
        std::ifstream file("src/cl/aplusb.cl");

        kernel_sources = std::string(
            std::istreambuf_iterator<char>(file),
            std::istreambuf_iterator<char>()
        );

        if (kernel_sources.size() == 0) {
            throw std::runtime_error(
                "Empty source file! May be you forgot to configure working directory properly?"
            );
        }

        // std::cout << kernel_sources << std::endl;
    }

    const char * program_text = kernel_sources.data();
    const std::size_t program_size = kernel_sources.size();

    const auto program = make_holder(
        clCreateProgramWithSource(
            *ctx,
            1,
            &program_text,
            &program_size,
            &result
        ),
        clReleaseProgram
    );

    OCL_SAFE_CALL(result);

    OCL_SAFE_CALL(clBuildProgram(
        *program,
        1,
        &device,
        nullptr,
        nullptr,
        nullptr
    ));

    {
        std::size_t log_size;
        OCL_SAFE_CALL(clGetProgramBuildInfo(
            *program,
            device,
            CL_PROGRAM_BUILD_LOG,
            0,
            nullptr,
            &log_size
        ));

        std::vector<char> log(log_size);
        OCL_SAFE_CALL(clGetProgramBuildInfo(
            *program,
            device,
            CL_PROGRAM_BUILD_LOG,
            log_size,
            log.data(),
            nullptr
        ));

        if (log_size > 1) {
            std::cout << "Log:" << std::endl;
            std::cout << log.data() << std::endl;
        }
    }

    const auto kernel = make_holder(
        clCreateKernel(*program, "aplusb", &result),
        clReleaseKernel
    );

    OCL_SAFE_CALL(result);

    {
        cl_uint i = 0;
        const cl_uint n_gpu = n;
        OCL_SAFE_CALL(clSetKernelArg(*kernel, i++, sizeof(cl_mem), &*as_buf));
        OCL_SAFE_CALL(clSetKernelArg(*kernel, i++, sizeof(cl_mem), &*bs_buf));
        OCL_SAFE_CALL(clSetKernelArg(*kernel, i++, sizeof(cl_mem), &*cs_buf));
        OCL_SAFE_CALL(clSetKernelArg(*kernel, i++, sizeof(cl_uint), &n_gpu));
    }

    {
        const size_t workGroupSize = 128;
        const size_t global_work_size = (n + workGroupSize - 1) / workGroupSize * workGroupSize;

        // Это вспомогательный секундомер, он замеряет время своего создания
        // и позволяет усреднять время нескольких замеров
        timer t;

        for (unsigned int i = 0; i < 20; ++i) {
            cl_event event;

            OCL_SAFE_CALL(clEnqueueNDRangeKernel(
                *cmd_queue,
                *kernel,
                1,
                nullptr,
                &global_work_size,
                &workGroupSize,
                0,
                nullptr,
                &event
            ));

            OCL_SAFE_CALL(clWaitForEvents(1, &event));

            clReleaseEvent(event);

            // При вызове nextLap секундомер запоминает текущий замер (текущий круг)
            // и начинает замерять время следующего круга
            t.nextLap();
        }

        // Среднее время круга (вычисления кернела) на самом деле считается не по всем замерам,
        // а лишь с 20%-перцентайля по 80%-перцентайль (как и стандартное отклонение)
        // подробнее об этом - см. timer.lapsFiltered

        std::cout << "Kernel average time: " << t.lapAvg()
                  << "+-" << t.lapStd()
                  << " s" << std::endl;

        std::cout << "GFlops: " << n / (t.lapAvg() * 1000000000) << std::endl;

        std::cout << "VRAM bandwidth: " <<  3 * n * sizeof(cl_float) / (t.lapAvg() * (1 << 30))
                  << " GB/s" << std::endl;
    }

    {
        timer t;
        for (unsigned int i = 0; i < 20; ++i) {
            OCL_SAFE_CALL(clEnqueueReadBuffer(
                *cmd_queue,
                *cs_buf,
                true,
                0,
                sizeof(cl_float) * n,
                cs.data(),
                0,
                nullptr,
                nullptr
            ));

            t.nextLap();
        }

        std::cout << "Result data transfer time: " << t.lapAvg()
                  << "+-" << t.lapStd()
                  << " s" << std::endl;

        std::cout << "VRAM -> RAM bandwidth: " << sizeof(cl_float) * n / (t.lapAvg() * (1 << 30))
                  << " GB/s" << std::endl;
    }

    for (unsigned int i = 0; i < n; ++i) {
        if (cs[i] != as[i] + bs[i]) {
            throw std::runtime_error("CPU and GPU results differ!");
        }
    }

    return 0;
}
