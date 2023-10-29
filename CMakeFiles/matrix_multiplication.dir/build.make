# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.22

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/ilia/GPGPUTasks2023

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/ilia/GPGPUTasks2023

# Include any dependencies generated for this target.
include CMakeFiles/matrix_multiplication.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/matrix_multiplication.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/matrix_multiplication.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/matrix_multiplication.dir/flags.make

src/cl/matrix_multiplication_cl.h: src/cl/matrix_multiplication.cl
src/cl/matrix_multiplication_cl.h: libs/gpu/hexdumparray
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --blue --bold --progress-dir=/home/ilia/GPGPUTasks2023/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Generating src/cl/matrix_multiplication_cl.h"
	libs/gpu/hexdumparray /home/ilia/GPGPUTasks2023/src/cl/matrix_multiplication.cl /home/ilia/GPGPUTasks2023/src/cl/matrix_multiplication_cl.h matrix_multiplication

CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.o: CMakeFiles/matrix_multiplication.dir/flags.make
CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.o: src/main_matrix_multiplication.cpp
CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.o: CMakeFiles/matrix_multiplication.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/ilia/GPGPUTasks2023/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.o -MF CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.o.d -o CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.o -c /home/ilia/GPGPUTasks2023/src/main_matrix_multiplication.cpp

CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/ilia/GPGPUTasks2023/src/main_matrix_multiplication.cpp > CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.i

CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/ilia/GPGPUTasks2023/src/main_matrix_multiplication.cpp -o CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.s

# Object files for target matrix_multiplication
matrix_multiplication_OBJECTS = \
"CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.o"

# External object files for target matrix_multiplication
matrix_multiplication_EXTERNAL_OBJECTS =

matrix_multiplication: CMakeFiles/matrix_multiplication.dir/src/main_matrix_multiplication.cpp.o
matrix_multiplication: CMakeFiles/matrix_multiplication.dir/build.make
matrix_multiplication: libs/clew/liblibclew.a
matrix_multiplication: libs/gpu/liblibgpu.a
matrix_multiplication: libs/utils/liblibutils.a
matrix_multiplication: libs/gpu/liblibgpu.a
matrix_multiplication: libs/utils/liblibutils.a
matrix_multiplication: libs/clew/liblibclew.a
matrix_multiplication: CMakeFiles/matrix_multiplication.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/ilia/GPGPUTasks2023/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable matrix_multiplication"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/matrix_multiplication.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/matrix_multiplication.dir/build: matrix_multiplication
.PHONY : CMakeFiles/matrix_multiplication.dir/build

CMakeFiles/matrix_multiplication.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/matrix_multiplication.dir/cmake_clean.cmake
.PHONY : CMakeFiles/matrix_multiplication.dir/clean

CMakeFiles/matrix_multiplication.dir/depend: src/cl/matrix_multiplication_cl.h
	cd /home/ilia/GPGPUTasks2023 && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/ilia/GPGPUTasks2023 /home/ilia/GPGPUTasks2023 /home/ilia/GPGPUTasks2023 /home/ilia/GPGPUTasks2023 /home/ilia/GPGPUTasks2023/CMakeFiles/matrix_multiplication.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/matrix_multiplication.dir/depend

