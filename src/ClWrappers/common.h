//
// Created by nik on 14.09.23.
//

#ifndef GPGPUTASKS2023_COMMON_H
#define GPGPUTASKS2023_COMMON_H

#include "CL/cl.h"
#include "libclew/ocl_init.h"
#include "libutils/fast_random.h"
#include "libutils/timer.h"

#include <cassert>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <vector>

template<typename T>
std::string to_string(T value) {
    std::ostringstream ss;
    ss << value;
    return ss.str();
}

inline void reportError(cl_int err, const std::string &filename, int line) {
    if (CL_SUCCESS == err)
        return;

    // Таблица с кодами ошибок:
    // libs/clew/CL/cl.h:103
    // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
    std::string message = "OpenCL error code " + to_string(err) + " encountered at " + filename + ":" + to_string(line);
    throw std::runtime_error(message);
}

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)


template<class R, class Function, class Id, class Param>
R GetValue(Function function, Id id, Param param) {
    R result;
    OCL_SAFE_CALL(function(id, param, sizeof(R), &result, nullptr));
    return result;
}

template<class R, class Size, class Function, typename... Args>
std::vector<R> GetVector(Function function, Args... args) {
    Size size = 0;
    OCL_SAFE_CALL(function(args...,  0, nullptr, &size));
    std::vector<R> result(size);
    OCL_SAFE_CALL(function(args..., size, result.data(), nullptr));
    return result;
}

#endif //GPGPUTASKS2023_COMMON_H
