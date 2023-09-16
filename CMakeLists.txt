cmake_minimum_required(VERSION 3.1)

add_subdirectory(libs)

project(aplusb)

set(CMAKE_CXX_STANDARD 11)
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

add_executable(${PROJECT_NAME} src/main.cpp)
target_link_libraries(${PROJECT_NAME} libclew libutils)
target_compile_definitions(${PROJECT_NAME} PUBLIC -DPROJECT_ROOT="${PROJECT_ROOT}")