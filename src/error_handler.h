#pragma once

#include <iostream>

namespace eh {
    template<typename T>
    void raiseFail(const T &a, const T &b, std::string message, std::string filename, int line) {
        if (a != b) {
            std::cerr << message << " But " << a << " != " << b << ", " << filename << ":" << line << std::endl;
            throw std::runtime_error(message);
        }
    }

#define EXPECT_THE_SAME(a, b, message) raiseFail(a, b, message, __FILE__, __LINE__)

}// namespace eh