cmake_minimum_required(VERSION 3.1)

add_subdirectory(libs)

project(aplusb)

set(HEADERS
        src/ClWrappers/common.h

        src/ClWrappers/Buffer.h
        src/ClWrappers/CommandQueue.h
        src/ClWrappers/Context.h
        src/ClWrappers/Device.h
        src/ClWrappers/Kernel.h
        src/ClWrappers/Program.h
        )

set(SOURCES
        src/main.cpp

        src/ClWrappers/Buffer.cpp
        src/ClWrappers/CommandQueue.cpp
        src/ClWrappers/Context.cpp
        src/ClWrappers/Device.cpp
        src/ClWrappers/Kernel.cpp
        src/ClWrappers/Program.cpp
        )

set(CMAKE_CXX_STANDARD 11)

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
target_link_libraries(${PROJECT_NAME} libclew libutils)
