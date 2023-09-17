#pragma once

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



template <typename R, typename SIZE_INFO_TYPE, typename CL_GET_INFO_F, typename... Args>
std::vector<R> getInfoVec(CL_GET_INFO_F getInfoF, Args... args) {
    SIZE_INFO_TYPE infoSize = 0;
    OCL_SAFE_CALL(getInfoF(args..., 0, nullptr, &infoSize));

    std::vector<R> infoVec(infoSize);
    OCL_SAFE_CALL(getInfoF(args..., infoSize, infoVec.data(), nullptr));
    return infoVec;
}

template <typename R, typename CL_GET_INFO_F, typename... Args>
R getInfo(CL_GET_INFO_F getInfoF, Args... args) {
    std::size_t infoSize = 0;
    OCL_SAFE_CALL(getInfoF(args..., 0, nullptr, &infoSize));

    R info = 0;
    OCL_SAFE_CALL(getInfoF(args..., infoSize, &info, nullptr));
    return info;
}