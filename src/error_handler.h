//
// Created by razerford on 12.09.23.
//

#ifndef APLUSB_ERROR_HANDLER_H
#define APLUSB_ERROR_HANDLER_H

#include <sstream>
#include <stdexcept>

namespace eh {

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
        // P.S. Быстрый переход к файлу в CLion: Ctrl+Shift+N -> cl.h (или даже с номером строки: cl.h:103) -> Enter
        std::string message =
                "OpenCL error code " +
                to_string(err) +
                " encountered at " +
                filename + ":" +
                to_string(line);

        throw std::runtime_error(message);
    }

    void reportErrorAndIgnore(cl_int err, const std::string &filename, int line, cl_int ignored) {
        if (err == ignored) {
            return;
        }
        reportError(err, filename, line);
    }

#define OCL_SAFE_CALL(expr) reportError(expr, __FILE__, __LINE__)
#define OCL_SAFE_CALL_IGNORE(expr, ignored) reportErrorAndIgnore(expr, __FILE__, __LINE__, ignored)
}

#endif //APLUSB_ERROR_HANDLER_H
