cmake_minimum_required(VERSION 3.1)

add_subdirectory(libs)

project(task5)

set(CMAKE_CXX_STANDARD 17)

# OpenMP позволит распараллеливать циклы на все ядра процессора простыми
# директивами

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS
      "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
else()
  message(WARNING "OpenMP not found!")
endif()

# convertIntoHeader CMake-функция объявлена в libs/gpu/CMakeLists.txt:71 Она
# считывает все байты из файла src/cl/aplusb.cl (т.е. весь исходный код кернела)
# и преобразует их в массив байтов в файле src/cl/aplusb_cl.h aplusb_kernel
# Обратите внимание что это происходит на этапе компиляции, кроме того
# необходимо чтобы файл src/cl/aplusb_cl.h был перечислен среди исходников для
# компиляции при вызове add_executable

convertintoheader(src/cl/radix.cl src/cl/radix_cl.h radix_kernel)
convertintoheader(src/cl/bitonic.cl src/cl/bitonic_cl.h bitonic_kernel)
convertintoheader(src/cl/merge.cl src/cl/merge_cl.h merge_kernel)
convertintoheader(src/cl/matrix_transpose.cl src/cl/matrix_transpose_cl.h
                  matrix_transpose_kernel)
convertintoheader(src/cl/prefix_sum.cl src/cl/prefix_sum_cl.h prefix_sum_kernel)
add_executable(
  radix src/main_radix.cpp src/cl/radix_cl.h src/cl/bitonic_cl.h
        src/cl/merge_cl.h src/cl/matrix_transpose_cl.h src/cl/prefix_sum_cl.h)
target_link_libraries(radix libclew libgpu libutils)
